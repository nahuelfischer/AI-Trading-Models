# Crop Data

All of my thoughts about sourcing, structuring and implementing crop data sets.

## Sources

I mainly used these websites to get the training data sets:
- U.S. Department of Agriculture: https://www.usda.gov/
-

### Sources for each crop

- Corn:
    - https://www.nass.usda.gov/Statistics_by_Subject/result.php?EE5F3F11-B8DC-3326-8759-6E1DCEC61E01&sector=CROPS&group=FIELD%20CROPS&comm=CORN
    - https://quickstats.nass.usda.gov/?source_desc=CENSUS#E1F47324-2591-33D3-91C6-7C4A160F4DA7
    - https://www.nass.usda.gov/AgCensus/archive/census_year/2012-census/index.html
    - https://www.nass.usda.gov/Charts_and_Maps/Crops_Cold_Storage/corn.php
    - https://www.nass.usda.gov/Charts_and_Maps/Crops_County/cr-pr.php
    - https://croplandcros.scinet.usda.gov/
    - https://www.agry.purdue.edu/ext/corn/news/timeless/yieldtrends.html
    - https://www.nass.usda.gov/Statistics_by_Subject/result.php?9F3F2601-4823-3230-8EB4-7512DC9E8829&sector=CROPS&group=FIELD%20CROPS&comm=CORN
    - https://www.ers.usda.gov/data-products/charts-of-note/chart-detail?chartId=76883
    - https://quickstats.nass.usda.gov/api
    - https://www.cropprophet.com/us-corn-production-by-state/
    - https://en.wikipedia.org/wiki/U.S._state

    After doing some research I concluded that it is most useful to use the corn yield (bushels per acre) as a train metric.
    The top corn-producing states are as of 2024:
    1. Iowa
    2. Illinois
    3. Nebraska
    4. Minnesota
    5. Indiana
    6. South Dakota
    7. Kansas
    8. Missouri
    9. Ohio
    10. North Dakota
    11. Wisconsin

    Therefore, I will train the model on weather data of these states, as they make up to more than 80% of US corn production.

- Soybeans:
    - sdaf

- xyz: