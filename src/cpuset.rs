//! CPU set parsing and validation utilities.
//!
//! Supports parsing CPU set strings in the format used by Linux taskset/numactl:
//! - Single CPUs: `"0"`, `"1"`, `"15"`
//! - Ranges: `"0-7"`, `"16-23"`
//! - Mixed: `"0-3,8-11"`, `"0,2,4,6"`

use crate::error::{LoomError, Result};

/// Parse a CPU set string into a sorted, deduplicated vector of CPU IDs.
///
/// # Format
///
/// The string format supports:
/// - Single CPU IDs: `"0"`, `"5"`
/// - Ranges (inclusive): `"0-7"`, `"16-23"`
/// - Comma-separated combinations: `"0-3,8-11"`, `"0,2,4,6-8"`
///
/// # Examples
///
/// ```
/// use loom_rs::cpuset::parse_cpuset;
///
/// let cpus = parse_cpuset("0-3,8-11").unwrap();
/// assert_eq!(cpus, vec![0, 1, 2, 3, 8, 9, 10, 11]);
///
/// let cpus = parse_cpuset("0,2,4").unwrap();
/// assert_eq!(cpus, vec![0, 2, 4]);
/// ```
///
/// # Errors
///
/// Returns `LoomError::InvalidCpuSet` if the string cannot be parsed.
pub fn parse_cpuset(s: &str) -> Result<Vec<usize>> {
    let s = s.trim();
    if s.is_empty() {
        return Err(LoomError::InvalidCpuSet("empty cpuset string".to_string()));
    }

    let mut cpus = Vec::new();

    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start.trim().parse().map_err(|_| {
                LoomError::InvalidCpuSet(format!("invalid range start in '{}'", part))
            })?;
            let end: usize = end.trim().parse().map_err(|_| {
                LoomError::InvalidCpuSet(format!("invalid range end in '{}'", part))
            })?;

            if start > end {
                return Err(LoomError::InvalidCpuSet(format!(
                    "range start {} > end {} in '{}'",
                    start, end, part
                )));
            }

            cpus.extend(start..=end);
        } else {
            let cpu: usize = part
                .parse()
                .map_err(|_| LoomError::InvalidCpuSet(format!("invalid CPU ID '{}'", part)))?;
            cpus.push(cpu);
        }
    }

    if cpus.is_empty() {
        return Err(LoomError::InvalidCpuSet(
            "no valid CPU IDs found".to_string(),
        ));
    }

    // Sort and deduplicate
    cpus.sort_unstable();
    cpus.dedup();

    Ok(cpus)
}

/// Get all logical CPU IDs available on this system.
///
/// Uses `core_affinity` to enumerate available CPUs.
///
/// # Examples
///
/// ```
/// use loom_rs::cpuset::available_cpus;
///
/// let cpus = available_cpus();
/// assert!(!cpus.is_empty());
/// ```
pub fn available_cpus() -> Vec<usize> {
    core_affinity::get_core_ids()
        .map(|ids| ids.into_iter().map(|id| id.id).collect())
        .unwrap_or_default()
}

/// Get the process CPU affinity mask.
///
/// On Linux, uses `sched_getaffinity(0)` to get the CPUs that the current
/// process is allowed to run on. This respects cgroups, containers, and
/// taskset restrictions.
///
/// On non-Linux platforms, returns `None` (caller should fall back to
/// `available_cpus()`).
///
/// # Examples
///
/// ```
/// use loom_rs::cpuset::get_process_affinity_mask;
///
/// if let Some(cpus) = get_process_affinity_mask() {
///     println!("Process allowed on CPUs: {:?}", cpus);
/// } else {
///     println!("Process affinity not available on this platform");
/// }
/// ```
#[cfg(target_os = "linux")]
pub fn get_process_affinity_mask() -> Option<Vec<usize>> {
    // cpu_set_t can hold up to 1024 CPUs by default
    // SAFETY: cpu_set_t is a plain data type safe to zero-initialize
    let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };

    // SAFETY: We're passing a valid pointer to a zeroed cpu_set_t
    // and the correct size. sched_getaffinity with pid=0 gets current process.
    unsafe {
        let result = libc::sched_getaffinity(
            0, // 0 = current process
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut cpuset,
        );

        if result != 0 {
            return None;
        }

        // Collect all CPUs that are set in the mask
        let mut cpus = Vec::new();
        // Check up to 1024 CPUs (the default cpu_set_t size)
        for cpu in 0..1024 {
            if libc::CPU_ISSET(cpu, &cpuset) {
                cpus.push(cpu);
            }
        }

        if cpus.is_empty() {
            None
        } else {
            Some(cpus)
        }
    }
}

/// Get the process CPU affinity mask.
///
/// On non-Linux platforms, returns `None` (caller should fall back to
/// `available_cpus()`).
#[cfg(not(target_os = "linux"))]
pub fn get_process_affinity_mask() -> Option<Vec<usize>> {
    None
}

/// Intersect two CPU sets.
///
/// Returns a sorted vector of CPU IDs that appear in both sets.
/// Uses HashSet for O(n) performance.
///
/// # Examples
///
/// ```
/// use loom_rs::cpuset::intersect_cpusets;
///
/// let a = vec![0, 1, 2, 3, 8, 9];
/// let b = vec![2, 3, 4, 5, 8];
/// let result = intersect_cpusets(&a, &b);
/// assert_eq!(result, vec![2, 3, 8]);
///
/// // Empty intersection
/// let result = intersect_cpusets(&[0, 1], &[2, 3]);
/// assert!(result.is_empty());
/// ```
pub fn intersect_cpusets(a: &[usize], b: &[usize]) -> Vec<usize> {
    use std::collections::HashSet;

    let set_a: HashSet<_> = a.iter().copied().collect();
    let set_b: HashSet<_> = b.iter().copied().collect();

    let mut result: Vec<_> = set_a.intersection(&set_b).copied().collect();
    result.sort_unstable();
    result
}

/// Validate that all CPUs in the set are available on this system.
///
/// # Errors
///
/// Returns `LoomError::CpuNotAvailable` if any CPU in the set is not available.
pub fn validate_cpuset(cpus: &[usize]) -> Result<()> {
    let available = available_cpus();
    for &cpu in cpus {
        if !available.contains(&cpu) {
            return Err(LoomError::CpuNotAvailable(cpu));
        }
    }
    Ok(())
}

/// Parse and validate a CPU set string against available system CPUs.
///
/// This is a convenience function that combines `parse_cpuset` and `validate_cpuset`.
///
/// # Errors
///
/// Returns an error if parsing fails or any CPU is not available.
pub fn parse_and_validate_cpuset(s: &str) -> Result<Vec<usize>> {
    let cpus = parse_cpuset(s)?;
    validate_cpuset(&cpus)?;
    Ok(cpus)
}

/// Format a slice of CPU IDs into a compact cpuset string.
///
/// This is the inverse of `parse_cpuset`. Consecutive CPU IDs are
/// collapsed into ranges.
///
/// # Examples
///
/// ```
/// use loom_rs::cpuset::format_cpuset;
///
/// assert_eq!(format_cpuset(&[0, 1, 2, 3]), "0-3");
/// assert_eq!(format_cpuset(&[0, 1, 2, 5, 6, 7]), "0-2,5-7");
/// assert_eq!(format_cpuset(&[0, 2, 4]), "0,2,4");
/// assert_eq!(format_cpuset(&[]), "");
/// ```
pub fn format_cpuset(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }

    // Sort and deduplicate
    let mut sorted: Vec<usize> = cpus.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut result = String::new();
    let mut i = 0;

    while i < sorted.len() {
        let start = sorted[i];
        let mut end = start;

        // Find the end of this consecutive range
        while i + 1 < sorted.len() && sorted[i + 1] == end + 1 {
            i += 1;
            end = sorted[i];
        }

        // Append separator if needed
        if !result.is_empty() {
            result.push(',');
        }

        // Format as range or single value
        if start == end {
            result.push_str(&start.to_string());
        } else {
            result.push_str(&format!("{}-{}", start, end));
        }

        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_cpu() {
        assert_eq!(parse_cpuset("0").unwrap(), vec![0]);
        assert_eq!(parse_cpuset("5").unwrap(), vec![5]);
        assert_eq!(parse_cpuset("15").unwrap(), vec![15]);
    }

    #[test]
    fn test_parse_range() {
        assert_eq!(parse_cpuset("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpuset("8-11").unwrap(), vec![8, 9, 10, 11]);
    }

    #[test]
    fn test_parse_mixed() {
        assert_eq!(
            parse_cpuset("0-3,8-11").unwrap(),
            vec![0, 1, 2, 3, 8, 9, 10, 11]
        );
        assert_eq!(parse_cpuset("0,2,4,6").unwrap(), vec![0, 2, 4, 6]);
        assert_eq!(parse_cpuset("0,2-4,8").unwrap(), vec![0, 2, 3, 4, 8]);
    }

    #[test]
    fn test_parse_with_whitespace() {
        assert_eq!(parse_cpuset(" 0-3 ").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpuset("0 - 3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpuset("0, 2, 4").unwrap(), vec![0, 2, 4]);
    }

    #[test]
    fn test_parse_deduplicates() {
        assert_eq!(parse_cpuset("0,0,1,1,2").unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_cpuset("0-2,1-3").unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_sorts() {
        assert_eq!(parse_cpuset("3,1,2,0").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(
            parse_cpuset("8-11,0-3").unwrap(),
            vec![0, 1, 2, 3, 8, 9, 10, 11]
        );
    }

    #[test]
    fn test_parse_empty_fails() {
        assert!(parse_cpuset("").is_err());
        assert!(parse_cpuset("   ").is_err());
    }

    #[test]
    fn test_parse_invalid_fails() {
        assert!(parse_cpuset("abc").is_err());
        assert!(parse_cpuset("0-abc").is_err());
        assert!(parse_cpuset("abc-5").is_err());
        assert!(parse_cpuset("-1").is_err()); // Negative numbers
    }

    #[test]
    fn test_parse_reversed_range_fails() {
        assert!(parse_cpuset("5-3").is_err());
    }

    #[test]
    fn test_available_cpus() {
        let cpus = available_cpus();
        // Should have at least one CPU
        assert!(!cpus.is_empty());
        // Should include CPU 0
        assert!(cpus.contains(&0));
    }

    #[test]
    fn test_validate_cpuset() {
        let available = available_cpus();
        // Validating available CPUs should succeed
        assert!(validate_cpuset(&available).is_ok());
        // Validating a very high CPU number should fail
        assert!(validate_cpuset(&[99999]).is_err());
    }

    #[test]
    fn test_format_cpuset_empty() {
        assert_eq!(format_cpuset(&[]), "");
    }

    #[test]
    fn test_format_cpuset_single() {
        assert_eq!(format_cpuset(&[0]), "0");
        assert_eq!(format_cpuset(&[5]), "5");
    }

    #[test]
    fn test_format_cpuset_range() {
        assert_eq!(format_cpuset(&[0, 1, 2, 3]), "0-3");
        assert_eq!(format_cpuset(&[8, 9, 10, 11]), "8-11");
    }

    #[test]
    fn test_format_cpuset_mixed() {
        assert_eq!(format_cpuset(&[0, 1, 2, 3, 8, 9, 10, 11]), "0-3,8-11");
        assert_eq!(format_cpuset(&[0, 2, 4, 6]), "0,2,4,6");
        assert_eq!(format_cpuset(&[0, 2, 3, 4, 8]), "0,2-4,8");
    }

    #[test]
    fn test_format_cpuset_unsorted() {
        // Should handle unsorted input
        assert_eq!(format_cpuset(&[3, 1, 2, 0]), "0-3");
        assert_eq!(format_cpuset(&[8, 9, 10, 11, 0, 1, 2, 3]), "0-3,8-11");
    }

    #[test]
    fn test_format_cpuset_duplicates() {
        // Should handle duplicates
        assert_eq!(format_cpuset(&[0, 0, 1, 1, 2]), "0-2");
    }

    #[test]
    fn test_format_parse_roundtrip() {
        // format -> parse should be identity
        let cases = vec![
            vec![0, 1, 2, 3],
            vec![0, 2, 4, 6],
            vec![0, 1, 2, 5, 6, 7, 10],
        ];
        for cpus in cases {
            let formatted = format_cpuset(&cpus);
            let parsed = parse_cpuset(&formatted).unwrap();
            assert_eq!(parsed, cpus);
        }
    }

    #[test]
    fn test_intersect_cpusets_overlap() {
        let a = vec![0, 1, 2, 3, 8, 9];
        let b = vec![2, 3, 4, 5, 8];
        let result = intersect_cpusets(&a, &b);
        assert_eq!(result, vec![2, 3, 8]);
    }

    #[test]
    fn test_intersect_cpusets_no_overlap() {
        let a = vec![0, 1, 2];
        let b = vec![3, 4, 5];
        let result = intersect_cpusets(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersect_cpusets_empty_input() {
        let a = vec![0, 1, 2];
        let result = intersect_cpusets(&a, &[]);
        assert!(result.is_empty());

        let result = intersect_cpusets(&[], &a);
        assert!(result.is_empty());

        let result = intersect_cpusets(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersect_cpusets_full_overlap() {
        let a = vec![0, 1, 2, 3];
        let b = vec![0, 1, 2, 3];
        let result = intersect_cpusets(&a, &b);
        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_intersect_cpusets_unsorted_input() {
        let a = vec![3, 1, 8, 2];
        let b = vec![9, 2, 1, 5];
        let result = intersect_cpusets(&a, &b);
        // Result should be sorted regardless of input order
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_intersect_cpusets_with_duplicates() {
        // Duplicates in input should be handled correctly
        let a = vec![0, 1, 1, 2, 2, 2];
        let b = vec![1, 2, 2, 3];
        let result = intersect_cpusets(&a, &b);
        assert_eq!(result, vec![1, 2]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_get_process_affinity_mask_returns_some() {
        // On Linux, this should always return Some
        let mask = get_process_affinity_mask();
        assert!(
            mask.is_some(),
            "get_process_affinity_mask should return Some on Linux"
        );
        let cpus = mask.unwrap();
        assert!(
            !cpus.is_empty(),
            "process should be allowed on at least one CPU"
        );
        // Should include CPU 0 in most cases
        assert!(cpus.contains(&0), "CPU 0 should typically be allowed");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_get_process_affinity_mask_subset_of_available() {
        let mask = get_process_affinity_mask().unwrap();
        let available = available_cpus();
        // Process affinity should be a subset of available CPUs
        for cpu in &mask {
            assert!(
                available.contains(cpu),
                "affinity mask CPU {} not in available CPUs",
                cpu
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_get_process_affinity_mask_returns_none() {
        // On non-Linux, this should return None
        let mask = get_process_affinity_mask();
        assert!(
            mask.is_none(),
            "get_process_affinity_mask should return None on non-Linux"
        );
    }
}
