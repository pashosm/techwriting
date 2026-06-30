# Order Re-Promising Macro

`Order_Repromising_Macro.xlsm` is a self-contained, macro-enabled Excel workbook
that re-promises a list of orders against the supply you actually have. Pick a
supply scenario, click one button, and every order gets a new promise date based
on when enough inventory has accumulated to fill it.

## Files

| File | Purpose |
|------|---------|
| `Order_Repromising_Macro.xlsm` | The workbook. Open in Excel and **Enable Content** to use the macro. |
| `Order_Repromising_Macro.bas`  | The VBA source for reference / re-import (identical to what's embedded). |

## How to use

1. Open the workbook in Excel and click **Enable Content** (it contains a macro).
2. Go to the **Control** sheet.
3. In cell **B2**, pick a supply scenario from the dropdown (`Base`, `High`, `Low`).
4. Click the **Run Re-Promise** button.
5. Results land on the **Orders** sheet in columns **D–F**, and a run summary
   appears on the Control sheet.

> The workbook ships with results already pre-computed for the **Base** scenario,
> so you can see it working even before enabling macros. Clicking **Run** simply
> recomputes for whatever scenario is selected.

## Sheets

### Orders
1,000 sample orders spread across 2027, **already sorted into priority order**
(earliest current promise date first). The macro processes them strictly
top-to-bottom — row order *is* the priority. To re-prioritize, reorder the rows.

| Column | Meaning |
|--------|---------|
| A `Order ID` | Order identifier |
| B `Current Promise Date` | The date currently promised (input) |
| C `Quantity` | Units required (input) |
| D `New Promise Date` | **Output** — the re-promised date |
| E `Status` | `Filled` or `Unfilled - End of Year` |
| F `Filled Qty` | Quantity consumed for this order |

### Supply
One row per week of 2027 (53 weeks). Column A is the week number, column B is the
**week-ending date** (the date inventory becomes available), and each remaining
column is a **supply scenario**.

| Column | Meaning |
|--------|---------|
| A `Week #` | Week of the year |
| B `Week Ending Date` | Last date in that week (the "supply available" date) |
| C `Base` / D `High` / E `Low` | Quantity available that week, per scenario |

### Control
The dropdown (B2), the **Run Re-Promise** button, and the last-run summary
(scenario, orders processed, filled, pushed to end-of-year, quantity consumed,
leftover inventory, run timestamp).

## The re-promising logic

- A **running inventory balance** starts at 0. Each week's supply is **added** to
  it (unused supply carries forward — it accumulates).
- Orders are taken in list order. For each order, weeks are consumed (added to
  inventory) until the running balance is **≥** the order quantity.
- When an order is filled, its quantity is removed from inventory and its
  **new promise date** is the **later of**:
  - the supply-available date (the week-ending date that satisfied it), and
  - its current promise date.

  (So if supply is available *after* the current promise, the date slips to the
  supply date; if supply is already available *before* it, the current promise
  date is kept.)
- If the running balance is below the order quantity, no inventory is consumed and
  the engine pulls in the **next week** of supply, repeating until it's enough.
- When supply for the year is **exhausted**, the current order and every order
  after it are pushed to the **last day of the year (31-Dec-2027)**.

## Choosing supply columns / adding scenarios

Only the **supply** column is selectable at run time (via the B2 dropdown), per
the agreed design — demand columns are fixed and orders are processed in row
order. To add another supply scenario, just:

1. Add a new column on the **Supply** sheet: put the scenario name in row 1 and
   the weekly quantities below it.

That's it — the **B2 dropdown updates automatically**. It's driven by a dynamic
named range (`ScenarioList` = `OFFSET(Supply!$C$1,0,0,1,COUNTA(Supply!$1:$1)-2)`)
that spans every header from column C onward, so any number of contiguous
scenario columns is picked up with no dropdown editing. The macro then matches
the selected name against the Supply header row to find the right column.

## Sample data at a glance

With the bundled sample data (total demand ≈ 275,958 units):

| Scenario | Total supply | Orders filled | Pushed to EOY |
|----------|-------------:|--------------:|--------------:|
| Base | ~248,000 | 897 | 103 |
| High | ~331,000 | 1,000 | 0 |
| Low  | ~166,000 | 599 | 401 |
