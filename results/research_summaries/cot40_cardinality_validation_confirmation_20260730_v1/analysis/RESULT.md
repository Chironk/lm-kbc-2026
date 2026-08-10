# Exact-support + cardinality validation confirmation

The selector and guard were frozen using training labels only. The incumbent pipeline has prior validation-selected lineage, so this is a development confirmation rather than a blind-test claim.

- Baseline F1: **0.518450**
- New F1: **0.514467**
- Delta: **-0.003983**
- Changed/helped/harmed/neutral: **49/13/16/20**
- `awardWonBy` is identity-only because exact validation Qwen generation artifacts do not exist for those ten rows.

| relation | baseline | support+cardinality | delta |
|---|---:|---:|---:|
| awardWonBy | 0.114372 | 0.114372 | +0.000000 |
| companyTradesAtStockExchange | 0.742905 | 0.716667 | -0.026238 |
| countryLandBordersCountry | 0.961540 | 0.972127 | +0.010587 |
| hasArea | 0.420000 | 0.420000 | +0.000000 |
| hasCapacity | 0.180000 | 0.180000 | +0.000000 |
| personHasCityOfDeath | 0.470000 | 0.470000 | +0.000000 |
