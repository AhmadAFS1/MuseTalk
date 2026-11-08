<!-- ni30-final-cost-receipt-v1 -->
# Completed validation rental: cost and cleanup

Instance **59107834** (`mtboot-ni30-r1-20261104a`) is verified absent, with its expiry timer **inactive**. It used one stock RTX 3090, one create, pinned worker revision `4e2b02e`, a null template hash, and a lifetime cap of four hours. No quality pass is implied by artifact retrieval; the archive's actual job status was **ok**.

| Estimate component | USD |
|---|---:|
| Earlier rounds 1 + 2 | $1.722680 |
| Validation rental time (3.578010 hours; disk included) | $0.935438 |
| Validation rental metered transfer | $0.013366 |
| AWS transport allowance, rounded up | $0.29 |
| **Combined estimate** | **$2.961485** |
| Budget cap | $10.00 |

The local result archive is 291,880,960 bytes and matches SHA256 `6b794b6da0223b09580fefd18d80b82f4bf55dad368de7869166dfe84eae10cc`. Deletion and absence are recorded for the task's input archive, probe, and result archive. Versioned objects require exact-version deletion; an unversioned result requires an explicit disabled-versioning proof.

AWS allowance: $0.25 for input transport, plus 291880960/1,000,000,000 × $0.09 for one effective output download, plus $0.01 for requests/storage, rounded upward to the next cent. [AWS S3 pricing](https://aws.amazon.com/s3/pricing/); [us-east-1 storage/request basis](https://docs.aws.amazon.com/solutions/latest/live-streaming-on-aws-with-amazon-s3/cost-example-1.html).

**Estimated, not invoice-confirmed; tax excluded.** No free-tier credit is assumed. Container counters may omit host/image traffic; extra/retried downloads or retained versions can add charges. The recorded evidence supports an estimate below $10, rather than an exact posted bill.
