Attribute VB_Name = "Module1"
Option Explicit

' ============================================================
'  Order Re-Promising Engine
'  Consumes available supply (running inventory) week by week and
'  assigns each order a new promise date based on when enough
'  inventory has accumulated.  Orders are processed TOP-DOWN in the
'  order they appear on the Orders sheet (that row order is the
'  priority).  Pick which supply scenario to consume from on the
'  Control sheet, then click "Run Re-Promise".
' ============================================================

Public Sub RunRepromise()
    Dim wsO As Worksheet, wsS As Worksheet, wsC As Worksheet
    On Error Resume Next
    Set wsO = ThisWorkbook.Worksheets("Orders")
    Set wsS = ThisWorkbook.Worksheets("Supply")
    Set wsC = ThisWorkbook.Worksheets("Control")
    On Error GoTo 0
    If wsO Is Nothing Or wsS Is Nothing Or wsC Is Nothing Then
        MsgBox "Could not find the Orders, Supply and Control sheets.", vbExclamation
        Exit Sub
    End If

    Dim scenario As String
    scenario = Trim(CStr(wsC.Range("B2").Value))
    If Len(scenario) = 0 Then
        MsgBox "Please choose a supply scenario in the Control sheet (cell B2).", vbExclamation
        Exit Sub
    End If

    ' --- Locate the supply column whose header matches the chosen scenario ---
    Dim sHdrLastCol As Long, c As Long, supCol As Long
    sHdrLastCol = wsS.Cells(1, wsS.Columns.Count).End(xlToLeft).Column
    supCol = 0
    For c = 3 To sHdrLastCol
        If Trim(CStr(wsS.Cells(1, c).Value)) = scenario Then
            supCol = c
            Exit For
        End If
    Next c
    If supCol = 0 Then
        MsgBox "Supply scenario '" & scenario & "' was not found in the Supply header row.", vbExclamation
        Exit Sub
    End If

    ' --- Read supply weeks into arrays (col B = week ending date, supCol = qty) ---
    Dim sLastRow As Long
    sLastRow = wsS.Cells(wsS.Rows.Count, 1).End(xlUp).Row
    Dim nWeeks As Long
    nWeeks = sLastRow - 1
    If nWeeks < 1 Then
        MsgBox "No supply rows were found on the Supply sheet.", vbExclamation
        Exit Sub
    End If

    Dim wkDate() As Double, wkQty() As Double
    ReDim wkDate(1 To nWeeks)
    ReDim wkQty(1 To nWeeks)
    Dim arrDate As Variant, arrQty As Variant
    arrDate = wsS.Range(wsS.Cells(2, 2), wsS.Cells(sLastRow, 2)).Value
    arrQty = wsS.Range(wsS.Cells(2, supCol), wsS.Cells(sLastRow, supCol)).Value
    Dim i As Long
    For i = 1 To nWeeks
        wkDate(i) = CDbl(arrDate(i, 1))
        If IsNumeric(arrQty(i, 1)) Then wkQty(i) = CDbl(arrQty(i, 1)) Else wkQty(i) = 0#
    Next i

    ' --- Read orders (col B = current promise date, col C = quantity) ---
    Dim oLastRow As Long
    oLastRow = wsO.Cells(wsO.Rows.Count, 1).End(xlUp).Row
    Dim nOrders As Long
    nOrders = oLastRow - 1
    If nOrders < 1 Then
        MsgBox "No orders were found on the Orders sheet.", vbExclamation
        Exit Sub
    End If
    Dim oPromise As Variant, oQty As Variant
    oPromise = wsO.Range(wsO.Cells(2, 2), wsO.Cells(oLastRow, 2)).Value
    oQty = wsO.Range(wsO.Cells(2, 3), wsO.Cells(oLastRow, 3)).Value

    ' --- Re-promising algorithm ---
    Dim eoy As Double
    eoy = CDbl(DateSerial(2027, 12, 31))
    Dim resDate() As Variant, resStat() As Variant, resFill() As Variant
    ReDim resDate(1 To nOrders, 1 To 1)
    ReDim resStat(1 To nOrders, 1 To 1)
    ReDim resFill(1 To nOrders, 1 To 1)

    Dim inv As Double:           inv = 0#
    Dim wk As Long:              wk = 1
    Dim curSupplyDate As Double: curSupplyDate = 0#
    Dim haveSupplyDate As Boolean: haveSupplyDate = False
    Dim qty As Double, promise As Double, newDate As Double
    Dim filledCount As Long, eoyCount As Long, totalConsumed As Double

    For i = 1 To nOrders
        If IsNumeric(oQty(i, 1)) Then qty = CDbl(oQty(i, 1)) Else qty = 0#
        promise = CDbl(oPromise(i, 1))

        ' Accumulate weekly supply into the running inventory until it is
        ' large enough for this order, or until we run out of supply weeks.
        Do While inv < qty And wk <= nWeeks
            inv = inv + wkQty(wk)
            curSupplyDate = wkDate(wk)
            haveSupplyDate = True
            wk = wk + 1
        Loop

        If inv >= qty Then
            inv = inv - qty
            totalConsumed = totalConsumed + qty
            ' New promise = later of (supply available date, current promise date)
            If haveSupplyDate And curSupplyDate > promise Then
                newDate = curSupplyDate
            Else
                newDate = promise
            End If
            resDate(i, 1) = newDate
            resStat(i, 1) = "Filled"
            resFill(i, 1) = qty
            filledCount = filledCount + 1
        Else
            ' Supply exhausted - this and all remaining orders go to end of year
            resDate(i, 1) = eoy
            resStat(i, 1) = "Unfilled - End of Year"
            resFill(i, 1) = 0#
            eoyCount = eoyCount + 1
        End If
    Next i

    ' --- Write results back to Orders sheet (cols D, E, F) ---
    Application.ScreenUpdating = False
    With wsO.Range(wsO.Cells(2, 4), wsO.Cells(oLastRow, 4))
        .Value = resDate
        .NumberFormat = "yyyy-mm-dd"
    End With
    wsO.Range(wsO.Cells(2, 5), wsO.Cells(oLastRow, 5)).Value = resStat
    wsO.Range(wsO.Cells(2, 6), wsO.Cells(oLastRow, 6)).Value = resFill
    Application.ScreenUpdating = True

    ' --- Summary on the Control sheet ---
    wsC.Range("B5").Value = scenario
    wsC.Range("B6").Value = nOrders
    wsC.Range("B7").Value = filledCount
    wsC.Range("B8").Value = eoyCount
    wsC.Range("B9").Value = totalConsumed
    wsC.Range("B10").Value = inv
    wsC.Range("B11").Value = Now
    wsC.Range("B11").NumberFormat = "yyyy-mm-dd hh:mm:ss"

    MsgBox "Re-promise complete." & vbCrLf & vbCrLf & _
           "Scenario      : " & scenario & vbCrLf & _
           "Orders        : " & nOrders & vbCrLf & _
           "Filled        : " & filledCount & vbCrLf & _
           "Pushed to EOY : " & eoyCount & vbCrLf & _
           "Qty consumed  : " & Format(totalConsumed, "#,##0") & vbCrLf & _
           "Leftover inv  : " & Format(inv, "#,##0"), vbInformation, "Order Re-Promising"
End Sub
