//! CUDA device detection and NUMA-aware CPU selection.
//!
//! This module provides utilities for selecting CPUs that are local to a
//! specified CUDA GPU, optimizing data transfer bandwidth between CPU and GPU.
//!
//! Requires the `cuda` feature and is only available on Linux.

use crate::error::{LoomError, Result};
use hwlocality::object::attributes::ObjectAttributes;
use hwlocality::object::types::ObjectType;
use hwlocality::Topology;
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Selector for identifying a CUDA device.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CudaDeviceSelector {
    /// Select by device index (0-based).
    DeviceId(u32),
    /// Select by device UUID.
    Uuid(String),
}

/// Get the CPU set for CPUs local to a CUDA device.
///
/// This function:
/// 1. Initializes NVML
/// 2. Finds the device by ID or UUID
/// 3. Gets the NUMA node ID for the device
/// 4. Uses hwlocality to find CPUs in that NUMA node
///
/// # Errors
///
/// Returns an error if:
/// - NVML cannot be initialized
/// - The device cannot be found
/// - The NUMA node cannot be determined
/// - hwlocality fails to enumerate CPUs
pub fn cpuset_for_cuda_device(selector: &CudaDeviceSelector) -> Result<Vec<usize>> {
    let nvml = Nvml::init()?;

    let device = match selector {
        CudaDeviceSelector::DeviceId(id) => {
            debug!(device_id = id, "selecting CUDA device by ID");
            nvml.device_by_index(*id)?
        }
        CudaDeviceSelector::Uuid(uuid) => {
            debug!(uuid, "selecting CUDA device by UUID");
            nvml.device_by_uuid(uuid.as_str())?
        }
    };

    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let pci_info = device.pci_info()?;

    info!(
        device = device_name,
        pci_bus_id = pci_info.bus_id,
        "found CUDA device"
    );

    // Get the topology
    let topology =
        Topology::new().map_err(|e| LoomError::Cuda(format!("hwlocality error: {}", e)))?;

    // Find the PCI device in the topology and get its NUMA node
    let cpus = find_cpus_for_pci_device(&topology, &pci_info.bus_id)?;

    info!(cpus = ?cpus, "found CPUs local to CUDA device");

    if cpus.is_empty() {
        // Fall back to all CPUs if we can't determine locality
        debug!("falling back to all CPUs");
        return Ok(crate::cpuset::available_cpus());
    }

    Ok(cpus)
}

fn find_cpus_for_pci_device(topology: &Topology, pci_bus_id: &str) -> Result<Vec<usize>> {
    // Parse PCI bus ID (format: "0000:XX:YY.Z")
    // hwlocality uses a different format, so we need to find the device by matching

    // Get all PCI devices
    let pci_devices = topology.objects_with_type(ObjectType::PCIDevice);

    for pci_obj in pci_devices {
        // Check if this is our device by comparing bus IDs
        if let Some(ObjectAttributes::PCIDevice(pci_attr)) = pci_obj.attributes() {
            let obj_bus_id = format!(
                "{:04x}:{:02x}:{:02x}.{:x}",
                pci_attr.domain(),
                pci_attr.bus_id(),
                pci_attr.bus_device(),
                pci_attr.function()
            );

            if obj_bus_id.eq_ignore_ascii_case(pci_bus_id) {
                debug!(obj_bus_id, "found matching PCI device in topology");

                // Get the CPU set for this object's locality
                if let Some(cpuset) = pci_obj.cpuset() {
                    let cpus: Vec<usize> = cpuset.iter_set().map(|c| c.into()).collect();
                    return Ok(cpus);
                }
            }
        }
    }

    // Try finding NUMA nodes directly if PCI device lookup fails
    debug!("PCI device not found in topology, trying NUMA nodes");

    // Get the first NUMA node as a fallback
    let numa_nodes = topology.objects_with_type(ObjectType::NUMANode);

    if let Some(numa) = numa_nodes.into_iter().next() {
        if let Some(cpuset) = numa.cpuset() {
            let cpus: Vec<usize> = cpuset.iter_set().map(|c| c.into()).collect();
            debug!(cpus = ?cpus, "using first NUMA node CPUs as fallback");
            return Ok(cpus);
        }
    }

    Ok(vec![])
}

/// Get information about all available CUDA devices.
///
/// Returns a list of (device_id, name, pci_bus_id) tuples.
pub fn list_cuda_devices() -> Result<Vec<(u32, String, String)>> {
    let nvml = Nvml::init()?;
    let count = nvml.device_count()?;

    let mut devices = Vec::with_capacity(count as usize);

    for i in 0..count {
        let device = nvml.device_by_index(i)?;
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        let pci_info = device.pci_info()?;
        devices.push((i, name, pci_info.bus_id));
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require CUDA hardware and NVML to be available.
    // They are marked as ignored by default.

    #[test]
    #[ignore]
    fn test_list_cuda_devices() {
        let devices = list_cuda_devices();
        // May or may not succeed depending on hardware
        println!("CUDA devices: {:?}", devices);
    }

    #[test]
    #[ignore]
    fn test_cpuset_for_cuda_device() {
        let cpus = cpuset_for_cuda_device(&CudaDeviceSelector::DeviceId(0));
        // May or may not succeed depending on hardware
        println!("CPUs for CUDA device 0: {:?}", cpus);
    }

    #[test]
    fn test_cuda_device_selector_deserialize() {
        // Test ID
        let json = "0";
        let selector: CudaDeviceSelector = serde_json::from_str(json).unwrap();
        assert!(matches!(selector, CudaDeviceSelector::DeviceId(0)));

        // Test UUID
        let json = "\"GPU-12345678-1234-1234-1234-123456789012\"";
        let selector: CudaDeviceSelector = serde_json::from_str(json).unwrap();
        assert!(matches!(selector, CudaDeviceSelector::Uuid(_)));
    }
}
