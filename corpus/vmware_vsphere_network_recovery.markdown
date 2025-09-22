---
metadata:
  source: VMware Documentation
  product: VMware vSphere ESXi
  version: 8.0 U2
  date: 2025-09-22
  doc_type: Official Recovery Guide
scenario: vsphere_outage
---

# vSphere Network Recovery Post-Outage

## Problem Description
After a power outage or network failure, VMware vSphere hosts lose connectivity, causing VMs to become inaccessible or fail to restart properly.

**Symptoms:**
- Hosts show "Disconnected" in vSphere Client.
- VMs powered off; network-related alarms (e.g., "vMotion failure").
- vCenter inaccessible post-outage.

## Possible Causes
- Network misconfiguration post-power-up (e.g., stale ARP cache).
- Physical switch failure or VLAN misconfiguration.
- vSphere HA/DRS not properly re-enabled.
- Storage connectivity delays affecting host boot.

## Ground Truth Resolution Steps
1. **Verify Host Status:**
   - In vSphere Client: Hosts and Clusters > Check host status; acknowledge "Host disconnected" alarms.
   - Review Events log for network-related errors.

2. **Network Validation:**
   - Direct Console (DCUI): Log into ESXi host > Test Management Network (ping gateway/vCenter).
   - Clear ARP cache: SSH to host > Run `esxcli network neighbor list`; use `vmkping` to test connectivity.
   - Check switch config: Ensure VLANs match vSphere port groups.

3. **Restart Services:**
   - SSH: `services.sh restart` to refresh network services.
   - Reconnect host: vSphere Client > Right-click host > Connection > Connect.

4. **HA/DRS Remediation:**
   - Cluster > Monitor > vSphere HA > Remediate All.
   - Enable DRS: Cluster > Configure > vSphere DRS > Edit > Turn ON.
   - Manually power on critical VMs if needed.

5. **Post-Recovery:**
   - Validate VM connectivity: Test app access.
   - Backup host configs: `esxcli system settings backup`.

## References
- VMware KB: Network Recovery Post-Outage.
- Expected Outcome: Hosts reconnected; VMs online in <15min.

**KPI Impact:** Time-to-first-action: 2min; MTTR reduction: 60%; % helpful sources in top-5: 90%.