---
metadata:
  source: Okta Support Documentation
  product: Okta Verify MFA
  version: 2.15
  date: 2025-09-22
  doc_type: Official Troubleshooting Guide
scenario: okta_mfa_failure
---

# Okta Verify or SMS MFA Troubleshooting Guide

## Problem Description
Users experience MFA failures during login, such as push notifications not arriving or SMS codes not delivering, leading to authentication blocks. This is common in remote access scenarios, impacting executive productivity and security compliance.

**Symptoms:**
- Okta Verify app push fails to prompt or times out.
- SMS code not received or invalid after entry.
- Error: "Invalid passcode" or "Push notification failed."
- Intermittent issues on mobile devices after OS updates.

## Possible Causes
- Network connectivity issues (WiFi/mobile data) preventing push delivery.
- Device clock sync problems with Okta servers.
- App permissions denied (notifications, background refresh).
- Carrier/SMS gateway delays or blacklisting.
- Account enrollment issues or expired factors.

## Ground Truth Resolution Steps
1. **User-Side Basic Checks:**
   - Verify device connected to internet; toggle WiFi/mobile data.
   - Ensure Okta Verify app updated to latest version; force close and reopen.
   - Check app permissions: Enable notifications and background app refresh in device settings.

2. **Clock Synchronization:**
   - On iOS: Settings > General > Date & Time > Set Automatically.
   - On Android: Settings > System > Date & Time > Use network-provided time.
   - Restart device and retry MFA prompt.

3. **SMS-Specific Fixes:**
   - Confirm phone number correct in Okta profile; request resend.
   - Test SMS from another service (e.g., bank) to rule out carrier issues.
   - If delayed: Wait 5-10min; enter code manually if received late.

4. **Admin-Side Verification (IT Help Desk):**
   - In Okta Admin Console: Directory > People > User > Security > Factors > Review enrolled factors; reset if needed.
   - Check system logs for errors: Reports > System Log > Filter by user/event time.
   - Test enrollment: Have user re-enroll via okta.com > Security > MFA > Add Factor.

5. **Advanced Recovery:**
   - If locked out: Use backup code or admin recovery (Admin > Security > Recovery).
   - Escalate to Okta Support with logs if persists.

## References
- Okta Verify App Logs: Enable debug mode in app settings for detailed errors.
- Expected Outcome: Successful MFA completion; user access restored in <5min.

**KPI Impact:** Time-to-first-action: 3min; MTTR reduction: 45%; % helpful sources in top-5: 80%.