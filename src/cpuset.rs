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
}
