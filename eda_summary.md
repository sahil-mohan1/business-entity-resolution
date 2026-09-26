# Stage 1 EDA Summary

`
======================================================================
  STAGE 1 EDA REPORT
  Generated: 2026-09-26 10:13:26
======================================================================

======================================================================
  1. FILE OVERVIEW
======================================================================
  train_source1             200.3 MB     2,206,821 rows
  train_source2             466.6 MB     5,034,616 rows
  train_source3             480.4 MB     5,285,603 rows
  train_gt                  121.1 MB     2,206,821 rows
  test_source1              166.9 MB     1,732,544 rows
  test_source2              485.9 MB     4,887,273 rows
  test_source3              482.6 MB     5,082,316 rows

======================================================================
  2. COLUMN NAMES
======================================================================

  [train_source1]  ['entity_id', 'business_name', 'business_address', 'country']

  [train_source2]  ['entity_id', 'business_name', 'business_address', 'country']

  [train_source3]  ['entity_id', 'business_name', 'business_address', 'country']

  [train_gt]  ['source1_entity_id', 'matched_entity_ids']

  [test_source1]  ['entity_id', 'business_name', 'business_address', 'country']

  [test_source2]  ['entity_id', 'business_name', 'business_address', 'country']

  [test_source3]  ['entity_id', 'business_name', 'business_address', 'country']

======================================================================
  3. MISSING VALUES
======================================================================

  [train_source1]
    entity_id                        0.00%
    business_name                    0.00%
    business_address                 0.00%
    country                          0.00%

  [train_source2]
    entity_id                        0.00%
    business_name                    0.00%
    business_address                 3.36%
    country                          0.00%

  [train_source3]
    entity_id                        0.00%
    business_name                    0.00%
    business_address                 3.48%
    country                          0.00%

  [test_source1]
    entity_id                        0.00%
    business_name                    0.00%
    business_address                 0.00%
    country                          0.00%

  [test_source2]
    entity_id                        0.00%
    business_name                    0.00%
    business_address                 2.66%
    country                          0.00%

  [test_source3]
    entity_id                        0.00%
    business_name                    0.00%
    business_address                 2.76%
    country                          0.00%

======================================================================
  4. GROUND TRUTH ANALYSIS
======================================================================

  GT columns: ['source1_entity_id', 'matched_entity_ids']
  GT rows: 2,206,821
  S1 col: ['source1_entity_id']  S2 col: []  S3 col: []

  Unique S1 with >=1 match: 2,206,821
  Total GT rows: 2,206,821

  Match-count distribution:
      1 match(es) -> 2,206,821  ##############################

  Max=1  Mean=1.000  Median=1.0

  S1 total: 2,206,821  matched: 2,206,821  singletons: 0 (0.0%)

======================================================================
  5. COUNTRY DISTRIBUTION
======================================================================

  [train_source1] sample=50,000
    US                              29,965  (59.9%)
    India                           20,035  (40.1%)

  [train_source2] sample=50,000
    US                              30,097  (60.2%)
    India                           19,903  (39.8%)

  [train_source3] sample=50,000
    US                              29,790  (59.6%)
    India                           20,210  (40.4%)

======================================================================
  6. NAME AND ADDRESS LENGTH DISTRIBUTIONS
======================================================================

  [train_source1]
    name_length  mean:24.1 median:24 min:3 max:71 std:7.7
    name_tokens  mean:3.6 median:4 max:12
    addr_length  mean:52.0 median:41 min:16 max:201
    addr_tokens  mean:8.0 median:7

  [train_source2]
    name_length  mean:25.0 median:25 min:2 max:104 std:8.9
    name_tokens  mean:3.5 median:4 max:15
    addr_length  mean:47.7 median:37 min:12 max:176
    addr_tokens  mean:7.5 median:6

  [train_source3]
    name_length  mean:25.2 median:25 min:2 max:80 std:9.5
    name_tokens  mean:3.5 median:4 max:13
    addr_length  mean:48.5 median:42 min:6 max:198
    addr_tokens  mean:7.4 median:6

======================================================================
  7. NOISE PROFILE - TOKEN OVERLAP ON MATCHED PAIRS
======================================================================
  Skipped.

======================================================================
  8. ENCODING AND CHARACTER ISSUES
======================================================================

  [train_source1] non-ASCII: 0/50,000 (0.0%)

  [train_source2] non-ASCII: 7,447/50,000 (14.9%)
    e.g.: '\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921'
    e.g.: '\u0906\u0926\u093f\u0924\u094d\u092f \u092a\u094d\u0930\u0949\u092a\u0930\u094d\u091f\u0940\u091c \u090f\u0932\u090f\u0932\u092a\u0940'
    e.g.: '\u0938\u0928 \u0915\u0902\u0938\u094d\u091f\u094d\u0930\u0915\u094d\u0936\u0902\u0938 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921'

  [train_source3] non-ASCII: 5,840/50,000 (11.7%)
    e.g.: 'LLC Moncada L\xe9arning Center'
    e.g.: 'B\xe9que'
    e.g.: '\u0b85\u0bb0\u0bbf\u0bb9\u0ba8\u0bcd\u0ba4\u0bcd Foundation Private Limited'

======================================================================
  9. NAIVE BASELINE F0.5
======================================================================

  Strategy A - Predict NOTHING:   F0.5=0.000 (recall=0)
  Strategy B - Predict EVERYTHING: F0.5~0.000 (precision~0)

  Hypothetical targets:
    P=0.95 R=0.90 -> F0.5=0.9396
    P=0.90 R=0.90 -> F0.5=0.9000
    P=0.80 R=0.90 -> F0.5=0.8182
    P=0.95 R=0.70 -> F0.5=0.8867

======================================================================
  10. SAMPLE RECORDS
======================================================================

  [train_source1]
  Row 0:
    entity_id: S1-925783039
    business_name: Orelee's Barbershop
    business_address: 1795 Westchester Drive, High Point, NC
    country: US

  Row 1:
    entity_id: S1-773889195
    business_name: Prime Money
    business_address: 17560 Ellis Road, Tahlequah, OK
    country: US

  Row 2:
    entity_id: S1-377745466
    business_name: B+ Retail Inc
    business_address: 1712 Montebello Avenue, Phoenix, AZ
    country: US


  [train_source2]
  Row 0:
    entity_id: S2-166376419
    business_name: राम मार्केटिंग प्राइवेट लिमिटेड
    business_address: KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi
    country: India

  Row 1:
    entity_id: S2-764573417
    business_name: -- Holloway Peak Inc Seafood
    business_address: 105 ELM ST, MORGANTON, NC
    country: US

  Row 2:
    entity_id: S2-639257739
    business_name: आदित्य प्रॉपर्टीज एलएलपी
    business_address: G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh
    country: India


  [train_source3]
  Row 0:
    entity_id: S3-202863386
    business_name: wilfordhancock.com
    business_address: Mack Rd, Haltom City, Texas
    country: US

  Row 1:
    entity_id: S3-859268022
    business_name: International South Consultants Private Ltd
    business_address: nan
    country: India

  Row 2:
    entity_id: S3-22467283
    business_name: LLC Moncada Léarning Center
    business_address: 5780 Fawn Ct, Fort Worth, Texas
    country: US

`
