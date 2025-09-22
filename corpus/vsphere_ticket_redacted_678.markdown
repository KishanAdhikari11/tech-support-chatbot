---
metadata:
  source: Fabricated Incident Ticket
  product: VMware vSphere ESXi
  version: N/A
  date: 2025-09-22
  doc_type: Redacted Ticket Example
scenario: vsphere_outage
---

# Ticket #VSP-678: VMs Down After Data Center Power Outage

## Summary
Post-power outage, multiple VMs in vSphere cluster are offline, and hosts show as disconnected in vSphere Client.

**Symptoms:**
- Hosts marked "Not Responding."
- VMs powered off; HA failover did not occur.
- vCenter UI slow or inaccessible.

## Possible Causes
- Power sequencing issue (storage offline before hosts).
- Network disruption post-outage.
- HA configuration stale.

## Ground Truth Resolution Steps
1. Verify power to storage arrays; ensure online before hosts.
2. Power on hosts via DCUI; wait 5min for full boot.
3. Reconnect hosts: vSphere Client > Right-click host > Connection > Connect.
4. Check HA settings: Cluster > Configure > vSphere Availability > Ensure Host Monitoring enabled.
5. Manually start critical VMs; run DRS recommendations.

## References
- Internal vSphere recovery runbook; VMware HA Guide.
- Expected Outcome: Cluster restored; VMs online in 12min.

**KPI Impact:** MTTR reduction: 65% (12min vs. 35min); time-to-first-action: 1min.