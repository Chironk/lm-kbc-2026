# Portable unified validation replay

The historical validation-subject capacity ledger is excluded.
Validation labels were opened only after predictions were frozen.

- portable pooled macro-F1: **0.518636**
- historical strict macro-F1: **0.520729**
- pooled delta: **-0.002092**
- portable capacity F1: **0.1700**
- historical capacity F1: **0.1800**

| relation | portable | historical strict | delta |
|---|---:|---:|---:|
| awardWonBy | 0.114372 | 0.114372 | +0.000000 |
| companyTradesAtStockExchange | 0.749905 | 0.749905 | +0.000000 |
| countryLandBordersCountry | 0.967265 | 0.967265 | +0.000000 |
| hasArea | 0.420000 | 0.420000 | +0.000000 |
| hasCapacity | 0.170000 | 0.180000 | -0.010000 |
| personHasCityOfDeath | 0.470000 | 0.470000 | +0.000000 |

## Changed rows by stage

- awardWonBy: component_surface_residual=9, ministral_cot40_two_thirds=3
- companyTradesAtStockExchange: component_surface_residual=4, ministral_cot40_two_thirds=5, strict_symbolic_graph=6
- countryLandBordersCountry: ministral_cot40_two_thirds=4, strict_symbolic_graph=4
- hasArea: area_unanimous_new_component=5, ministral_cot40_two_thirds=12
- hasCapacity: ministral_cot40_two_thirds=1
- personHasCityOfDeath: ministral_cot40_two_thirds=13
