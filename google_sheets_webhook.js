/**
 * =============================================================================
 * TOUCHLESS SMART ATTENDANCE KIOSK - GOOGLE APPS SCRIPT WEBHOOK
 * =============================================================================
 * Receives JSON POST payloads from the attendance kiosk background sync thread
 * and appends rows directly to the active Google Spreadsheet.
 *
 * EXPANDED COLUMNS APPENDED:
 *   [Date, In-Time, Roll Number, Student Name, Year, Branch, Status, Synced At (Cloud)]
 *
 * SETUP INSTRUCTIONS:
 * 1. Open Google Sheets (https://sheets.new).
 * 2. Rename Sheet1 to "Attendance" (or keep default).
 * 3. Go to: Extensions -> Apps Script.
 * 4. Delete any code in Code.gs and paste this entire file.
 * 5. Click "Deploy" (top right) -> "New deployment".
 * 6. Click the gear icon next to "Select type" -> choose "Web app".
 * 7. Configure:
 *      - Description: "Attendance Kiosk Webhook"
 *      - Execute as: "Me" (your email)
 *      - Who has access: "Anyone" (CRITICAL for non-authenticated kiosk requests)
 * 8. Click "Deploy", review/grant permissions.
 * 9. Copy the "Web app URL" (ends in /exec).
 * 10. Paste the URL into `app.py` as `GOOGLE_SHEETS_WEBHOOK_URL`.
 * =============================================================================
 */

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    lock.waitLock(10000);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({
      status: "error",
      message: "Server busy: lock timeout."
    })).setMimeType(ContentService.MimeType.JSON);
  }

  try {
    if (!e || !e.postData || !e.postData.contents) {
      return ContentService.createTextOutput(JSON.stringify({
        status: "error",
        message: "No post data received."
      })).setMimeType(ContentService.MimeType.JSON);
    }

    var payload;
    try {
      payload = JSON.parse(e.postData.contents);
    } catch (parseError) {
      return ContentService.createTextOutput(JSON.stringify({
        status: "error",
        message: "Invalid JSON format: " + parseError.toString()
      })).setMimeType(ContentService.MimeType.JSON);
    }

    var items = [];
    if (payload.records && Array.isArray(payload.records)) {
      items = payload.records;
    } else if (payload.roll_number !== undefined || payload.student_id !== undefined) {
      items = [payload];
    } else if (Array.isArray(payload)) {
      items = payload;
    }

    if (items.length === 0) {
      return ContentService.createTextOutput(JSON.stringify({
        status: "error",
        message: "Payload contained 0 attendance records."
      })).setMimeType(ContentService.MimeType.JSON);
    }

    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = ss.getActiveSheet();

    // Auto-initialize or auto-upgrade expanded header row with all 8 academic fields
    var headers = [
      "Date",
      "In-Time",
      "Roll Number",
      "Student Name",
      "Year",
      "Branch",
      "Status",
      "Synced At (Cloud)"
    ];

    if (sheet.getLastRow() === 0) {
      sheet.appendRow(headers);
      var headerRange = sheet.getRange(1, 1, 1, headers.length);
      headerRange.setFontWeight("bold");
      headerRange.setBackground("#1e293b");
      headerRange.setFontColor("#FFFFFF");
      sheet.setFrozenRows(1);
    } else {
      // If the sheet already exists with old headers, automatically upgrade Row 1
      var currentFirstRow = sheet.getRange(1, 1, 1, Math.max(sheet.getLastColumn(), headers.length)).getValues()[0];
      var headerStr = currentFirstRow.join(" ");
      if (headerStr.indexOf("Roll Number") === -1 || headerStr.indexOf("Branch") === -1) {
        sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        var headerRange = sheet.getRange(1, 1, 1, headers.length);
        headerRange.setFontWeight("bold");
        headerRange.setBackground("#1e293b");
        headerRange.setFontColor("#FFFFFF");
        sheet.setFrozenRows(1);
      }
    }

    var nowIso = new Date().toISOString();
    var rowsToAppend = [];

    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      var datePart = item.date || "";
      var inTimePart = item.in_time || "";
      var rawTimestamp = item.timestamp || "";

      // Fallback date and time parsing
      if (!datePart || !inTimePart) {
        if (rawTimestamp && rawTimestamp.indexOf(" ") !== -1) {
          var parts = rawTimestamp.split(" ");
          datePart = datePart || parts[0];
          inTimePart = inTimePart || parts[1];
        } else {
          var d = new Date();
          datePart = datePart || Utilities.formatDate(d, Session.getScriptTimeZone(), "yyyy-MM-dd");
          inTimePart = inTimePart || Utilities.formatDate(d, Session.getScriptTimeZone(), "HH:mm:ss");
        }
      }

      var rollNumber = item.roll_number || item.student_id || "N/A";
      var studentName = item.student_name || "Unknown";
      var year = item.year || "N/A";
      var branch = item.branch || "N/A";
      var status = item.status || "PRESENT";

      rowsToAppend.push([
        datePart,
        inTimePart,
        rollNumber,
        studentName,
        year,
        branch,
        status,
        nowIso
      ]);
    }

    if (rowsToAppend.length > 0) {
      var startRow = sheet.getLastRow() + 1;
      var numRows = rowsToAppend.length;
      var numCols = rowsToAppend[0].length;
      sheet.getRange(startRow, 1, numRows, numCols).setValues(rowsToAppend);

      // Auto-resize columns for clean readability
      for (var col = 1; col <= numCols; col++) {
        sheet.autoResizeColumn(col);
      }
    }

    return ContentService.createTextOutput(JSON.stringify({
      status: "success",
      rows_written: rowsToAppend.length,
      synced_at: nowIso
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (error) {
    return ContentService.createTextOutput(JSON.stringify({
      status: "error",
      message: error.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  return ContentService.createTextOutput(JSON.stringify({
    status: "online",
    service: "Touchless Smart Attendance Kiosk Webhook",
    columns: ["Date", "In-Time", "Roll Number", "Student Name", "Year", "Branch", "Status", "Synced At"]
  })).setMimeType(ContentService.MimeType.JSON);
}
