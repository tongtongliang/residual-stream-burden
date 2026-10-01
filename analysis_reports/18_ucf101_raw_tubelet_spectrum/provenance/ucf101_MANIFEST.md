# UCF101 Local Manifest

Downloaded and extracted on 2026-08-16 from the official UCF CRCV endpoints.

## Files

| Item | Local path | Size / count |
|---|---|---:|
| Original video archive | `archives/UCF101.rar` | 6,932,971,618 bytes |
| Official recognition splits | `archives/UCF101TrainTestSplits-RecognitionTask.zip` | 113,943 bytes |
| Extracted videos | `videos/UCF-101/<class>/*.avi` | 13,320 videos, 101 classes |
| Extracted video bytes | `videos/UCF-101/` | 7,193,326,396 bytes |

## Official split audit

| Split | Train | Test | Train/test overlap | Missing files | Coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,537 | 3,783 | 0 | 0 | 13,320 |
| 2 | 9,586 | 3,734 | 0 | 0 | 13,320 |
| 3 | 9,624 | 3,696 | 0 | 0 | 13,320 |

`classInd.txt` contains 101 entries. Every path referenced by every official
split exists in the extracted video tree.
