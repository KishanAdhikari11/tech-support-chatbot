---
metadata:
  source: Cisco Community Forum
  product: Cisco AnyConnect Secure Mobility Client
  version: 4.10
  date: 2023-12-05
  doc_type: Community Troubleshooting Thread
scenario: vpn_outage
---

# Cisco AnyConnect "Authentication Failed" Error Troubleshooting

## Problem Description
Users report an "Authentication Failed" error when attempting to connect to the corporate VPN using Cisco AnyConnect at the Windows login screen. This prevents remote access and login to work resources. The issue appears isolated to specific user profiles or devices, while others connect without issues.

**Symptoms:**
- Error message: "Authentication Failed"
- Connection attempt fails after entering credentials.
- Temporary workaround: Using a hardware token allows connection, but it's not feasible for daily use.

## Possible Causes
- Mandatory hardware token requirement for authentication (e.g., certificate-based).
- Security policy violation when attempting token-less connection.
- Misconfiguration on the VPN headend (ASA or similar).
- User account issues (locked, inactive) or incorrect group selection.

## Ground Truth Resolution Steps
1. **Verify Credentials and Token Policy:**
   - Confirm username, password, and group are entered correctly.
   - Check if the VPN policy requires a hardware token (e.g., RSA SecurID). If so, contact IT to confirm if exceptions can be made for your profile.

2. **Check Account Status:**
   - Ensure the user account is active and not locked out in Active Directory or the authentication server (RADIUS/LDAP).

3. **Gather Logs for IT Review:**
   - In AnyConnect, click the "Diagnostics" button to generate logs.
   - Share logs with IT, who should review VPN headend debugs during a connection attempt.

4. **IT-Side Troubleshooting:**
   - On the VPN headend: Enable debug for authentication (e.g., `debug aaa authentication` on ASA).
   - Verify reachability to authentication servers.
   - Adjust policy if needed: Allow certificate-based auth without token for specific users/devices.

## References
- Cisco AnyConnect Logs: Review for specific errors like "AAA failed" or "Certificate mismatch."
- Expected Outcome: Successful connection without token if policy adjusted; otherwise, use token as interim.

**KPI Impact:** Reduces time-to-first-action from 30min to 5min via quick log collection; estimated MTTR reduction: 50%.