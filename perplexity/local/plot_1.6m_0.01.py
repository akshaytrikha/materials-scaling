import re
import pandas as pd

log_data = '''
Data Iteration:   0%|                                     | 0/1 [00:00<?, ?it/s]
Model is on device cuda and has 202612 parameters

Epoch Progress:   0%|                                  | 0/1000 [00:00<?, ?it/s]Dataset Size: 1%, Epoch: 1, Train Loss: 10.993632197380066, Val Loss: 10.987699706833084

Epoch Progress:   0%|                          | 1/1000 [00:01<26:41,  1.60s/it]
Epoch Progress:   0%|                          | 2/1000 [00:02<22:32,  1.35s/it]
Epoch Progress:   0%|                          | 3/1000 [00:03<21:11,  1.27s/it]
Epoch Progress:   0%|                          | 4/1000 [00:05<20:33,  1.24s/it]
Epoch Progress:   0%|▏                         | 5/1000 [00:06<20:11,  1.22s/it]
Epoch Progress:   1%|▏                         | 6/1000 [00:07<19:57,  1.20s/it]
Epoch Progress:   1%|▏                         | 7/1000 [00:08<19:49,  1.20s/it]
Epoch Progress:   1%|▏                         | 8/1000 [00:09<19:43,  1.19s/it]
Epoch Progress:   1%|▏                         | 9/1000 [00:11<19:40,  1.19s/it]
Epoch Progress:   1%|▎                        | 10/1000 [00:12<19:38,  1.19s/it]Dataset Size: 1%, Epoch: 11, Train Loss: 10.98400616645813, Val Loss: 10.981350997825722

Epoch Progress:   1%|▎                        | 11/1000 [00:13<19:34,  1.19s/it]
Epoch Progress:   1%|▎                        | 12/1000 [00:14<19:30,  1.18s/it]
Epoch Progress:   1%|▎                        | 13/1000 [00:15<19:27,  1.18s/it]
Epoch Progress:   1%|▎                        | 14/1000 [00:16<19:25,  1.18s/it]
Epoch Progress:   2%|▍                        | 15/1000 [00:18<19:24,  1.18s/it]
Epoch Progress:   2%|▍                        | 16/1000 [00:19<19:22,  1.18s/it]
Epoch Progress:   2%|▍                        | 17/1000 [00:20<19:20,  1.18s/it]
Epoch Progress:   2%|▍                        | 18/1000 [00:21<19:20,  1.18s/it]
Epoch Progress:   2%|▍                        | 19/1000 [00:22<19:19,  1.18s/it]
Epoch Progress:   2%|▌                        | 20/1000 [00:24<19:17,  1.18s/it]Dataset Size: 1%, Epoch: 21, Train Loss: 10.96385669708252, Val Loss: 10.96678214878231

Epoch Progress:   2%|▌                        | 21/1000 [00:25<19:17,  1.18s/it]
Epoch Progress:   2%|▌                        | 22/1000 [00:26<19:15,  1.18s/it]
Epoch Progress:   2%|▌                        | 23/1000 [00:27<19:13,  1.18s/it]
Epoch Progress:   2%|▌                        | 24/1000 [00:28<19:14,  1.18s/it]
Epoch Progress:   2%|▋                        | 25/1000 [00:29<19:12,  1.18s/it]
Epoch Progress:   3%|▋                        | 26/1000 [00:31<19:18,  1.19s/it]
Epoch Progress:   3%|▋                        | 27/1000 [00:32<19:19,  1.19s/it]
Epoch Progress:   3%|▋                        | 28/1000 [00:33<19:15,  1.19s/it]
Epoch Progress:   3%|▋                        | 29/1000 [00:34<19:11,  1.19s/it]
Epoch Progress:   3%|▊                        | 30/1000 [00:35<19:08,  1.18s/it]Dataset Size: 1%, Epoch: 31, Train Loss: 10.931525349617004, Val Loss: 10.946058186617764

Epoch Progress:   3%|▊                        | 31/1000 [00:37<19:06,  1.18s/it]
Epoch Progress:   3%|▊                        | 32/1000 [00:38<19:04,  1.18s/it]
Epoch Progress:   3%|▊                        | 33/1000 [00:39<19:01,  1.18s/it]
Epoch Progress:   3%|▊                        | 34/1000 [00:40<19:00,  1.18s/it]
Epoch Progress:   4%|▉                        | 35/1000 [00:41<19:03,  1.18s/it]
Epoch Progress:   4%|▉                        | 36/1000 [00:43<19:46,  1.23s/it]
Epoch Progress:   4%|▉                        | 37/1000 [00:44<19:31,  1.22s/it]
Epoch Progress:   4%|▉                        | 38/1000 [00:45<19:28,  1.21s/it]
Epoch Progress:   4%|▉                        | 39/1000 [00:46<19:18,  1.21s/it]
Epoch Progress:   4%|█                        | 40/1000 [00:47<19:09,  1.20s/it]Dataset Size: 1%, Epoch: 41, Train Loss: 10.897677659988403, Val Loss: 10.922554734465363

Epoch Progress:   4%|█                        | 41/1000 [00:49<19:02,  1.19s/it]
Epoch Progress:   4%|█                        | 42/1000 [00:50<19:01,  1.19s/it]
Epoch Progress:   4%|█                        | 43/1000 [00:51<19:00,  1.19s/it]
Epoch Progress:   4%|█                        | 44/1000 [00:52<18:58,  1.19s/it]
Epoch Progress:   4%|█▏                       | 45/1000 [00:53<18:55,  1.19s/it]
Epoch Progress:   5%|█▏                       | 46/1000 [00:55<18:53,  1.19s/it]
Epoch Progress:   5%|█▏                       | 47/1000 [00:56<18:50,  1.19s/it]
Epoch Progress:   5%|█▏                       | 48/1000 [00:57<18:48,  1.19s/it]
Epoch Progress:   5%|█▏                       | 49/1000 [00:58<18:47,  1.19s/it]
Epoch Progress:   5%|█▎                       | 50/1000 [00:59<18:47,  1.19s/it]Dataset Size: 1%, Epoch: 51, Train Loss: 10.866418719291687, Val Loss: 10.899508847818746

Epoch Progress:   5%|█▎                       | 51/1000 [01:00<18:44,  1.19s/it]
Epoch Progress:   5%|█▎                       | 52/1000 [01:02<18:56,  1.20s/it]
Epoch Progress:   5%|█▎                       | 53/1000 [01:03<18:53,  1.20s/it]
Epoch Progress:   5%|█▎                       | 54/1000 [01:04<18:48,  1.19s/it]
Epoch Progress:   6%|█▍                       | 55/1000 [01:05<18:46,  1.19s/it]
Epoch Progress:   6%|█▍                       | 56/1000 [01:06<18:47,  1.19s/it]
Epoch Progress:   6%|█▍                       | 57/1000 [01:08<18:47,  1.20s/it]
Epoch Progress:   6%|█▍                       | 58/1000 [01:09<18:42,  1.19s/it]
Epoch Progress:   6%|█▍                       | 59/1000 [01:10<18:38,  1.19s/it]
Epoch Progress:   6%|█▌                       | 60/1000 [01:11<18:40,  1.19s/it]Dataset Size: 1%, Epoch: 61, Train Loss: 10.831833004951477, Val Loss: 10.878006551172826

Epoch Progress:   6%|█▌                       | 61/1000 [01:12<18:37,  1.19s/it]
Epoch Progress:   6%|█▌                       | 62/1000 [01:14<18:32,  1.19s/it]
Epoch Progress:   6%|█▌                       | 63/1000 [01:15<18:29,  1.18s/it]
Epoch Progress:   6%|█▌                       | 64/1000 [01:16<18:26,  1.18s/it]
Epoch Progress:   6%|█▋                       | 65/1000 [01:17<18:28,  1.19s/it]
Epoch Progress:   7%|█▋                       | 66/1000 [01:18<18:26,  1.18s/it]
Epoch Progress:   7%|█▋                       | 67/1000 [01:20<18:26,  1.19s/it]
Epoch Progress:   7%|█▋                       | 68/1000 [01:21<18:24,  1.18s/it]
Epoch Progress:   7%|█▋                       | 69/1000 [01:22<18:26,  1.19s/it]
Epoch Progress:   7%|█▊                       | 70/1000 [01:23<18:24,  1.19s/it]Dataset Size: 1%, Epoch: 71, Train Loss: 10.79837453365326, Val Loss: 10.859056683329793

Epoch Progress:   7%|█▊                       | 71/1000 [01:24<18:21,  1.19s/it]
Epoch Progress:   7%|█▊                       | 72/1000 [01:25<18:19,  1.19s/it]
Epoch Progress:   7%|█▊                       | 73/1000 [01:27<18:18,  1.18s/it]
Epoch Progress:   7%|█▊                       | 74/1000 [01:28<18:16,  1.18s/it]
Epoch Progress:   8%|█▉                       | 75/1000 [01:29<18:14,  1.18s/it]
Epoch Progress:   8%|█▉                       | 76/1000 [01:30<18:12,  1.18s/it]
Epoch Progress:   8%|█▉                       | 77/1000 [01:31<18:14,  1.19s/it]
Epoch Progress:   8%|█▉                       | 78/1000 [01:33<18:10,  1.18s/it]
Epoch Progress:   8%|█▉                       | 79/1000 [01:34<18:15,  1.19s/it]
Epoch Progress:   8%|██                       | 80/1000 [01:35<18:11,  1.19s/it]Dataset Size: 1%, Epoch: 81, Train Loss: 10.766895413398743, Val Loss: 10.842922346932548

Epoch Progress:   8%|██                       | 81/1000 [01:36<18:09,  1.19s/it]
Epoch Progress:   8%|██                       | 82/1000 [01:37<18:06,  1.18s/it]
Epoch Progress:   8%|██                       | 83/1000 [01:38<18:04,  1.18s/it]
Epoch Progress:   8%|██                       | 84/1000 [01:40<18:01,  1.18s/it]
Epoch Progress:   8%|██▏                      | 85/1000 [01:41<17:59,  1.18s/it]
Epoch Progress:   9%|██▏                      | 86/1000 [01:42<18:02,  1.18s/it]
Epoch Progress:   9%|██▏                      | 87/1000 [01:43<18:00,  1.18s/it]
Epoch Progress:   9%|██▏                      | 88/1000 [01:44<17:57,  1.18s/it]
Epoch Progress:   9%|██▏                      | 89/1000 [01:46<17:56,  1.18s/it]
Epoch Progress:   9%|██▎                      | 90/1000 [01:47<17:55,  1.18s/it]Dataset Size: 1%, Epoch: 91, Train Loss: 10.728062868118286, Val Loss: 10.828242723043862

Epoch Progress:   9%|██▎                      | 91/1000 [01:48<17:53,  1.18s/it]
Epoch Progress:   9%|██▎                      | 92/1000 [01:49<17:51,  1.18s/it]
Epoch Progress:   9%|██▎                      | 93/1000 [01:50<17:50,  1.18s/it]
Epoch Progress:   9%|██▎                      | 94/1000 [01:51<17:53,  1.18s/it]
Epoch Progress:  10%|██▍                      | 95/1000 [01:53<17:52,  1.18s/it]
Epoch Progress:  10%|██▍                      | 96/1000 [01:54<17:49,  1.18s/it]
Epoch Progress:  10%|██▍                      | 97/1000 [01:55<17:47,  1.18s/it]
Epoch Progress:  10%|██▍                      | 98/1000 [01:56<17:45,  1.18s/it]
Epoch Progress:  10%|██▍                      | 99/1000 [01:57<17:44,  1.18s/it]
Epoch Progress:  10%|██▍                     | 100/1000 [01:59<17:43,  1.18s/it]Dataset Size: 1%, Epoch: 101, Train Loss: 10.684398531913757, Val Loss: 10.812971511444488

Epoch Progress:  10%|██▍                     | 101/1000 [02:00<17:42,  1.18s/it]
Epoch Progress:  10%|██▍                     | 102/1000 [02:01<17:41,  1.18s/it]
Epoch Progress:  10%|██▍                     | 103/1000 [02:02<17:43,  1.19s/it]
Epoch Progress:  10%|██▍                     | 104/1000 [02:03<17:43,  1.19s/it]
Epoch Progress:  10%|██▌                     | 105/1000 [02:05<17:47,  1.19s/it]
Epoch Progress:  11%|██▌                     | 106/1000 [02:06<17:48,  1.20s/it]
Epoch Progress:  11%|██▌                     | 107/1000 [02:07<17:43,  1.19s/it]
Epoch Progress:  11%|██▌                     | 108/1000 [02:08<17:40,  1.19s/it]
Epoch Progress:  11%|██▌                     | 109/1000 [02:09<17:37,  1.19s/it]
Epoch Progress:  11%|██▋                     | 110/1000 [02:10<17:37,  1.19s/it]Dataset Size: 1%, Epoch: 111, Train Loss: 10.642085671424866, Val Loss: 10.797246994910303

Epoch Progress:  11%|██▋                     | 111/1000 [02:12<17:37,  1.19s/it]
Epoch Progress:  11%|██▋                     | 112/1000 [02:13<17:33,  1.19s/it]
Epoch Progress:  11%|██▋                     | 113/1000 [02:14<17:30,  1.18s/it]
Epoch Progress:  11%|██▋                     | 114/1000 [02:15<17:27,  1.18s/it]
Epoch Progress:  12%|██▊                     | 115/1000 [02:16<17:25,  1.18s/it]
Epoch Progress:  12%|██▊                     | 116/1000 [02:18<17:24,  1.18s/it]
Epoch Progress:  12%|██▊                     | 117/1000 [02:19<17:23,  1.18s/it]
Epoch Progress:  12%|██▊                     | 118/1000 [02:20<17:24,  1.18s/it]
Epoch Progress:  12%|██▊                     | 119/1000 [02:21<17:25,  1.19s/it]
Epoch Progress:  12%|██▉                     | 120/1000 [02:22<17:25,  1.19s/it]Dataset Size: 1%, Epoch: 121, Train Loss: 10.576955199241638, Val Loss: 10.780165474136155

Epoch Progress:  12%|██▉                     | 121/1000 [02:23<17:25,  1.19s/it]
Epoch Progress:  12%|██▉                     | 122/1000 [02:25<17:25,  1.19s/it]
Epoch Progress:  12%|██▉                     | 123/1000 [02:26<17:22,  1.19s/it]
Epoch Progress:  12%|██▉                     | 124/1000 [02:27<17:18,  1.19s/it]
Epoch Progress:  12%|███                     | 125/1000 [02:28<17:16,  1.18s/it]
Epoch Progress:  13%|███                     | 126/1000 [02:29<17:14,  1.18s/it]
Epoch Progress:  13%|███                     | 127/1000 [02:31<17:14,  1.18s/it]
Epoch Progress:  13%|███                     | 128/1000 [02:32<17:15,  1.19s/it]
Epoch Progress:  13%|███                     | 129/1000 [02:33<17:13,  1.19s/it]
Epoch Progress:  13%|███                     | 130/1000 [02:34<17:10,  1.18s/it]Dataset Size: 1%, Epoch: 131, Train Loss: 10.510604858398438, Val Loss: 10.763404276463893

Epoch Progress:  13%|███▏                    | 131/1000 [02:35<17:08,  1.18s/it]
Epoch Progress:  13%|███▏                    | 132/1000 [02:37<17:10,  1.19s/it]
Epoch Progress:  13%|███▏                    | 133/1000 [02:38<17:09,  1.19s/it]
Epoch Progress:  13%|███▏                    | 134/1000 [02:39<17:05,  1.18s/it]
Epoch Progress:  14%|███▏                    | 135/1000 [02:40<17:03,  1.18s/it]
Epoch Progress:  14%|███▎                    | 136/1000 [02:41<17:03,  1.19s/it]
Epoch Progress:  14%|███▎                    | 137/1000 [02:42<17:00,  1.18s/it]
Epoch Progress:  14%|███▎                    | 138/1000 [02:44<16:57,  1.18s/it]
Epoch Progress:  14%|███▎                    | 139/1000 [02:45<16:56,  1.18s/it]
Epoch Progress:  14%|███▎                    | 140/1000 [02:46<16:55,  1.18s/it]Dataset Size: 1%, Epoch: 141, Train Loss: 10.430214643478394, Val Loss: 10.745892413250811

Epoch Progress:  14%|███▍                    | 141/1000 [02:47<16:55,  1.18s/it]
Epoch Progress:  14%|███▍                    | 142/1000 [02:48<16:52,  1.18s/it]
Epoch Progress:  14%|███▍                    | 143/1000 [02:50<16:52,  1.18s/it]
Epoch Progress:  14%|███▍                    | 144/1000 [02:51<16:50,  1.18s/it]
Epoch Progress:  14%|███▍                    | 145/1000 [02:52<16:52,  1.18s/it]
Epoch Progress:  15%|███▌                    | 146/1000 [02:53<16:50,  1.18s/it]
Epoch Progress:  15%|███▌                    | 147/1000 [02:54<16:48,  1.18s/it]
Epoch Progress:  15%|███▌                    | 148/1000 [02:55<16:46,  1.18s/it]
Epoch Progress:  15%|███▌                    | 149/1000 [02:57<16:44,  1.18s/it]
Epoch Progress:  15%|███▌                    | 150/1000 [02:58<16:43,  1.18s/it]Dataset Size: 1%, Epoch: 151, Train Loss: 10.328742861747742, Val Loss: 10.727804060106154

Epoch Progress:  15%|███▌                    | 151/1000 [02:59<16:41,  1.18s/it]
Epoch Progress:  15%|███▋                    | 152/1000 [03:00<16:41,  1.18s/it]
Epoch Progress:  15%|███▋                    | 153/1000 [03:01<16:44,  1.19s/it]
Epoch Progress:  15%|███▋                    | 154/1000 [03:03<16:41,  1.18s/it]
Epoch Progress:  16%|███▋                    | 155/1000 [03:04<16:38,  1.18s/it]
Epoch Progress:  16%|███▋                    | 156/1000 [03:05<16:36,  1.18s/it]
Epoch Progress:  16%|███▊                    | 157/1000 [03:06<16:35,  1.18s/it]
Epoch Progress:  16%|███▊                    | 158/1000 [03:07<16:39,  1.19s/it]
Epoch Progress:  16%|███▊                    | 159/1000 [03:08<16:37,  1.19s/it]
Epoch Progress:  16%|███▊                    | 160/1000 [03:10<16:35,  1.18s/it]Dataset Size: 1%, Epoch: 161, Train Loss: 10.259353160858154, Val Loss: 10.71531631420185

Epoch Progress:  16%|███▊                    | 161/1000 [03:11<16:33,  1.18s/it]
Epoch Progress:  16%|███▉                    | 162/1000 [03:12<16:32,  1.18s/it]
Epoch Progress:  16%|███▉                    | 163/1000 [03:13<16:29,  1.18s/it]
Epoch Progress:  16%|███▉                    | 164/1000 [03:14<16:27,  1.18s/it]
Epoch Progress:  16%|███▉                    | 165/1000 [03:16<16:25,  1.18s/it]
Epoch Progress:  17%|███▉                    | 166/1000 [03:17<16:23,  1.18s/it]
Epoch Progress:  17%|████                    | 167/1000 [03:18<16:23,  1.18s/it]
Epoch Progress:  17%|████                    | 168/1000 [03:19<16:21,  1.18s/it]
Epoch Progress:  17%|████                    | 169/1000 [03:20<16:20,  1.18s/it]
Epoch Progress:  17%|████                    | 170/1000 [03:21<16:22,  1.18s/it]Dataset Size: 1%, Epoch: 171, Train Loss: 10.14638364315033, Val Loss: 10.704420733761477

Epoch Progress:  17%|████                    | 171/1000 [03:23<16:19,  1.18s/it]
Epoch Progress:  17%|████▏                   | 172/1000 [03:24<16:16,  1.18s/it]
Epoch Progress:  17%|████▏                   | 173/1000 [03:25<16:15,  1.18s/it]
Epoch Progress:  17%|████▏                   | 174/1000 [03:26<16:14,  1.18s/it]
Epoch Progress:  18%|████▏                   | 175/1000 [03:27<16:11,  1.18s/it]
Epoch Progress:  18%|████▏                   | 176/1000 [03:29<16:11,  1.18s/it]
Epoch Progress:  18%|████▏                   | 177/1000 [03:30<16:09,  1.18s/it]
Epoch Progress:  18%|████▎                   | 178/1000 [03:31<16:08,  1.18s/it]
Epoch Progress:  18%|████▎                   | 179/1000 [03:32<16:11,  1.18s/it]
Epoch Progress:  18%|████▎                   | 180/1000 [03:33<16:08,  1.18s/it]Dataset Size: 1%, Epoch: 181, Train Loss: 10.047637104988098, Val Loss: 10.69870832368925

Epoch Progress:  18%|████▎                   | 181/1000 [03:34<16:07,  1.18s/it]
Epoch Progress:  18%|████▎                   | 182/1000 [03:36<16:05,  1.18s/it]
Epoch Progress:  18%|████▍                   | 183/1000 [03:37<16:03,  1.18s/it]
Epoch Progress:  18%|████▍                   | 184/1000 [03:38<16:02,  1.18s/it]
Epoch Progress:  18%|████▍                   | 185/1000 [03:39<16:09,  1.19s/it]
Epoch Progress:  19%|████▍                   | 186/1000 [03:40<16:05,  1.19s/it]
Epoch Progress:  19%|████▍                   | 187/1000 [03:42<16:04,  1.19s/it]
Epoch Progress:  19%|████▌                   | 188/1000 [03:43<16:00,  1.18s/it]
Epoch Progress:  19%|████▌                   | 189/1000 [03:44<15:57,  1.18s/it]
Epoch Progress:  19%|████▌                   | 190/1000 [03:45<15:54,  1.18s/it]Dataset Size: 1%, Epoch: 191, Train Loss: 9.957343578338623, Val Loss: 10.697462329616794

Epoch Progress:  19%|████▌                   | 191/1000 [03:46<15:53,  1.18s/it]
Epoch Progress:  19%|████▌                   | 192/1000 [03:47<15:53,  1.18s/it]
Epoch Progress:  19%|████▋                   | 193/1000 [03:49<15:51,  1.18s/it]
Epoch Progress:  19%|████▋                   | 194/1000 [03:50<15:49,  1.18s/it]
Epoch Progress:  20%|████▋                   | 195/1000 [03:51<15:50,  1.18s/it]
Epoch Progress:  20%|████▋                   | 196/1000 [03:52<15:49,  1.18s/it]
Epoch Progress:  20%|████▋                   | 197/1000 [03:53<15:46,  1.18s/it]
Epoch Progress:  20%|████▊                   | 198/1000 [03:54<15:44,  1.18s/it]
Epoch Progress:  20%|████▊                   | 199/1000 [03:56<15:43,  1.18s/it]
Epoch Progress:  20%|████▊                   | 200/1000 [03:57<15:41,  1.18s/it]Dataset Size: 1%, Epoch: 201, Train Loss: 9.849374413490295, Val Loss: 10.700371420228636

Epoch Progress:  20%|████▊                   | 201/1000 [03:58<15:40,  1.18s/it]
Epoch Progress:  20%|████▊                   | 202/1000 [03:59<15:40,  1.18s/it]
Epoch Progress:  20%|████▊                   | 203/1000 [04:00<15:39,  1.18s/it]
Epoch Progress:  20%|████▉                   | 204/1000 [04:02<15:39,  1.18s/it]
Epoch Progress:  20%|████▉                   | 205/1000 [04:03<15:38,  1.18s/it]
Epoch Progress:  21%|████▉                   | 206/1000 [04:04<15:36,  1.18s/it]
Epoch Progress:  21%|████▉                   | 207/1000 [04:05<15:34,  1.18s/it]
Epoch Progress:  21%|████▉                   | 208/1000 [04:06<15:33,  1.18s/it]
Epoch Progress:  21%|█████                   | 209/1000 [04:07<15:31,  1.18s/it]
Epoch Progress:  21%|█████                   | 210/1000 [04:09<15:29,  1.18s/it]Dataset Size: 1%, Epoch: 211, Train Loss: 9.790127754211426, Val Loss: 10.708391870771136

Epoch Progress:  21%|█████                   | 211/1000 [04:10<15:32,  1.18s/it]
Epoch Progress:  21%|█████                   | 212/1000 [04:11<15:31,  1.18s/it]
Epoch Progress:  21%|█████                   | 213/1000 [04:12<15:30,  1.18s/it]
Epoch Progress:  21%|█████▏                  | 214/1000 [04:13<15:28,  1.18s/it]
Epoch Progress:  22%|█████▏                  | 215/1000 [04:15<15:26,  1.18s/it]
Epoch Progress:  22%|█████▏                  | 216/1000 [04:16<15:24,  1.18s/it]
Epoch Progress:  22%|█████▏                  | 217/1000 [04:17<15:22,  1.18s/it]
Epoch Progress:  22%|█████▏                  | 218/1000 [04:18<15:21,  1.18s/it]
Epoch Progress:  22%|█████▎                  | 219/1000 [04:19<15:18,  1.18s/it]
Epoch Progress:  22%|█████▎                  | 220/1000 [04:20<15:16,  1.18s/it]Dataset Size: 1%, Epoch: 221, Train Loss: 9.735920786857605, Val Loss: 10.722757240394493

Epoch Progress:  22%|█████▎                  | 221/1000 [04:22<15:17,  1.18s/it]
Epoch Progress:  22%|█████▎                  | 222/1000 [04:23<15:15,  1.18s/it]
Epoch Progress:  22%|█████▎                  | 223/1000 [04:24<15:13,  1.18s/it]
Epoch Progress:  22%|█████▍                  | 224/1000 [04:25<15:11,  1.18s/it]
Epoch Progress:  22%|█████▍                  | 225/1000 [04:26<15:10,  1.18s/it]
Epoch Progress:  23%|█████▍                  | 226/1000 [04:27<15:09,  1.17s/it]
Epoch Progress:  23%|█████▍                  | 227/1000 [04:29<15:08,  1.17s/it]
Epoch Progress:  23%|█████▍                  | 228/1000 [04:30<15:06,  1.17s/it]
Epoch Progress:  23%|█████▍                  | 229/1000 [04:31<15:05,  1.18s/it]
Epoch Progress:  23%|█████▌                  | 230/1000 [04:32<15:05,  1.18s/it]Dataset Size: 1%, Epoch: 231, Train Loss: 9.67339563369751, Val Loss: 10.736796379089355

Epoch Progress:  23%|█████▌                  | 231/1000 [04:33<15:04,  1.18s/it]
Epoch Progress:  23%|█████▌                  | 232/1000 [04:35<15:04,  1.18s/it]
Epoch Progress:  23%|█████▌                  | 233/1000 [04:36<15:03,  1.18s/it]
Epoch Progress:  23%|█████▌                  | 234/1000 [04:37<15:02,  1.18s/it]
Epoch Progress:  24%|█████▋                  | 235/1000 [04:38<15:01,  1.18s/it]
Epoch Progress:  24%|█████▋                  | 236/1000 [04:39<15:00,  1.18s/it]
Epoch Progress:  24%|█████▋                  | 237/1000 [04:40<14:59,  1.18s/it]
Epoch Progress:  24%|█████▋                  | 238/1000 [04:42<15:05,  1.19s/it]
Epoch Progress:  24%|█████▋                  | 239/1000 [04:43<15:02,  1.19s/it]
Epoch Progress:  24%|█████▊                  | 240/1000 [04:44<14:58,  1.18s/it]Dataset Size: 1%, Epoch: 241, Train Loss: 9.624491333961487, Val Loss: 10.753132448567973

Epoch Progress:  24%|█████▊                  | 241/1000 [04:45<14:56,  1.18s/it]
Epoch Progress:  24%|█████▊                  | 242/1000 [04:46<14:54,  1.18s/it]
Epoch Progress:  24%|█████▊                  | 243/1000 [04:48<14:53,  1.18s/it]
Epoch Progress:  24%|█████▊                  | 244/1000 [04:49<14:51,  1.18s/it]
Epoch Progress:  24%|█████▉                  | 245/1000 [04:50<14:50,  1.18s/it]
Epoch Progress:  25%|█████▉                  | 246/1000 [04:51<14:51,  1.18s/it]
Epoch Progress:  25%|█████▉                  | 247/1000 [04:52<14:48,  1.18s/it]
Epoch Progress:  25%|█████▉                  | 248/1000 [04:53<14:46,  1.18s/it]
Epoch Progress:  25%|█████▉                  | 249/1000 [04:55<14:44,  1.18s/it]
Epoch Progress:  25%|██████                  | 250/1000 [04:56<14:43,  1.18s/it]Dataset Size: 1%, Epoch: 251, Train Loss: 9.580064415931702, Val Loss: 10.76960123978652

Epoch Progress:  25%|██████                  | 251/1000 [04:57<14:42,  1.18s/it]
Epoch Progress:  25%|██████                  | 252/1000 [04:58<14:41,  1.18s/it]
Epoch Progress:  25%|██████                  | 253/1000 [04:59<14:40,  1.18s/it]
Epoch Progress:  25%|██████                  | 254/1000 [05:00<14:38,  1.18s/it]
Epoch Progress:  26%|██████                  | 255/1000 [05:02<14:39,  1.18s/it]
Epoch Progress:  26%|██████▏                 | 256/1000 [05:03<14:37,  1.18s/it]
Epoch Progress:  26%|██████▏                 | 257/1000 [05:04<14:35,  1.18s/it]
Epoch Progress:  26%|██████▏                 | 258/1000 [05:05<14:34,  1.18s/it]
Epoch Progress:  26%|██████▏                 | 259/1000 [05:06<14:32,  1.18s/it]
Epoch Progress:  26%|██████▏                 | 260/1000 [05:08<14:32,  1.18s/it]Dataset Size: 1%, Epoch: 261, Train Loss: 9.554871201515198, Val Loss: 10.786719346975351

Epoch Progress:  26%|██████▎                 | 261/1000 [05:09<14:31,  1.18s/it]
Epoch Progress:  26%|██████▎                 | 262/1000 [05:10<14:29,  1.18s/it]
Epoch Progress:  26%|██████▎                 | 263/1000 [05:11<14:30,  1.18s/it]
Epoch Progress:  26%|██████▎                 | 264/1000 [05:12<14:31,  1.18s/it]
Epoch Progress:  26%|██████▎                 | 265/1000 [05:13<14:31,  1.19s/it]
Epoch Progress:  27%|██████▍                 | 266/1000 [05:15<14:28,  1.18s/it]
Epoch Progress:  27%|██████▍                 | 267/1000 [05:16<14:26,  1.18s/it]
Epoch Progress:  27%|██████▍                 | 268/1000 [05:17<14:25,  1.18s/it]
Epoch Progress:  27%|██████▍                 | 269/1000 [05:18<14:32,  1.19s/it]
Epoch Progress:  27%|██████▍                 | 270/1000 [05:19<14:26,  1.19s/it]Dataset Size: 1%, Epoch: 271, Train Loss: 9.512032628059387, Val Loss: 10.799213632360681

Epoch Progress:  27%|██████▌                 | 271/1000 [05:21<14:23,  1.18s/it]
Epoch Progress:  27%|██████▌                 | 272/1000 [05:22<14:23,  1.19s/it]
Epoch Progress:  27%|██████▌                 | 273/1000 [05:23<14:21,  1.18s/it]
Epoch Progress:  27%|██████▌                 | 274/1000 [05:24<14:18,  1.18s/it]
Epoch Progress:  28%|██████▌                 | 275/1000 [05:25<14:16,  1.18s/it]
Epoch Progress:  28%|██████▌                 | 276/1000 [05:27<14:13,  1.18s/it]
Epoch Progress:  28%|██████▋                 | 277/1000 [05:28<14:11,  1.18s/it]
Epoch Progress:  28%|██████▋                 | 278/1000 [05:29<14:09,  1.18s/it]
Epoch Progress:  28%|██████▋                 | 279/1000 [05:30<14:08,  1.18s/it]
Epoch Progress:  28%|██████▋                 | 280/1000 [05:31<14:09,  1.18s/it]Dataset Size: 1%, Epoch: 281, Train Loss: 9.490260243415833, Val Loss: 10.813965413477513

Epoch Progress:  28%|██████▋                 | 281/1000 [05:32<14:07,  1.18s/it]
Epoch Progress:  28%|██████▊                 | 282/1000 [05:34<14:05,  1.18s/it]
Epoch Progress:  28%|██████▊                 | 283/1000 [05:35<14:03,  1.18s/it]
Epoch Progress:  28%|██████▊                 | 284/1000 [05:36<14:02,  1.18s/it]
Epoch Progress:  28%|██████▊                 | 285/1000 [05:37<14:01,  1.18s/it]
Epoch Progress:  29%|██████▊                 | 286/1000 [05:38<14:00,  1.18s/it]
Epoch Progress:  29%|██████▉                 | 287/1000 [05:39<13:58,  1.18s/it]
Epoch Progress:  29%|██████▉                 | 288/1000 [05:41<13:59,  1.18s/it]
Epoch Progress:  29%|██████▉                 | 289/1000 [05:42<13:59,  1.18s/it]
Epoch Progress:  29%|██████▉                 | 290/1000 [05:43<13:57,  1.18s/it]Dataset Size: 1%, Epoch: 291, Train Loss: 9.456826448440552, Val Loss: 10.82813717482926

Epoch Progress:  29%|██████▉                 | 291/1000 [05:44<14:01,  1.19s/it]
Epoch Progress:  29%|███████                 | 292/1000 [05:45<13:57,  1.18s/it]
Epoch Progress:  29%|███████                 | 293/1000 [05:47<13:54,  1.18s/it]
Epoch Progress:  29%|███████                 | 294/1000 [05:48<13:54,  1.18s/it]
Epoch Progress:  30%|███████                 | 295/1000 [05:49<13:51,  1.18s/it]
Epoch Progress:  30%|███████                 | 296/1000 [05:50<13:49,  1.18s/it]
Epoch Progress:  30%|███████▏                | 297/1000 [05:51<13:50,  1.18s/it]
Epoch Progress:  30%|███████▏                | 298/1000 [05:52<13:47,  1.18s/it]
Epoch Progress:  30%|███████▏                | 299/1000 [05:54<13:46,  1.18s/it]
Epoch Progress:  30%|███████▏                | 300/1000 [05:55<13:44,  1.18s/it]Dataset Size: 1%, Epoch: 301, Train Loss: 9.440702199935913, Val Loss: 10.842520750962295

Epoch Progress:  30%|███████▏                | 301/1000 [05:56<13:43,  1.18s/it]
Epoch Progress:  30%|███████▏                | 302/1000 [05:57<13:41,  1.18s/it]
Epoch Progress:  30%|███████▎                | 303/1000 [05:58<13:41,  1.18s/it]
Epoch Progress:  30%|███████▎                | 304/1000 [06:00<13:40,  1.18s/it]
Epoch Progress:  30%|███████▎                | 305/1000 [06:01<13:38,  1.18s/it]
Epoch Progress:  31%|███████▎                | 306/1000 [06:02<13:38,  1.18s/it]
Epoch Progress:  31%|███████▎                | 307/1000 [06:03<13:37,  1.18s/it]
Epoch Progress:  31%|███████▍                | 308/1000 [06:04<13:35,  1.18s/it]
Epoch Progress:  31%|███████▍                | 309/1000 [06:05<13:33,  1.18s/it]
Epoch Progress:  31%|███████▍                | 310/1000 [06:07<13:31,  1.18s/it]Dataset Size: 1%, Epoch: 311, Train Loss: 9.399873971939087, Val Loss: 10.853972348299893

Epoch Progress:  31%|███████▍                | 311/1000 [06:08<13:29,  1.18s/it]
Epoch Progress:  31%|███████▍                | 312/1000 [06:09<13:29,  1.18s/it]
Epoch Progress:  31%|███████▌                | 313/1000 [06:10<13:28,  1.18s/it]
Epoch Progress:  31%|███████▌                | 314/1000 [06:11<13:29,  1.18s/it]
Epoch Progress:  32%|███████▌                | 315/1000 [06:12<13:28,  1.18s/it]
Epoch Progress:  32%|███████▌                | 316/1000 [06:14<13:26,  1.18s/it]
Epoch Progress:  32%|███████▌                | 317/1000 [06:15<13:24,  1.18s/it]
Epoch Progress:  32%|███████▋                | 318/1000 [06:16<13:26,  1.18s/it]
Epoch Progress:  32%|███████▋                | 319/1000 [06:17<13:24,  1.18s/it]
Epoch Progress:  32%|███████▋                | 320/1000 [06:18<13:22,  1.18s/it]Dataset Size: 1%, Epoch: 321, Train Loss: 9.395278215408325, Val Loss: 10.865112738175826

Epoch Progress:  32%|███████▋                | 321/1000 [06:20<13:20,  1.18s/it]
Epoch Progress:  32%|███████▋                | 322/1000 [06:21<13:18,  1.18s/it]
Epoch Progress:  32%|███████▊                | 323/1000 [06:22<13:22,  1.19s/it]
Epoch Progress:  32%|███████▊                | 324/1000 [06:23<13:19,  1.18s/it]
Epoch Progress:  32%|███████▊                | 325/1000 [06:24<13:18,  1.18s/it]
Epoch Progress:  33%|███████▊                | 326/1000 [06:25<13:16,  1.18s/it]
Epoch Progress:  33%|███████▊                | 327/1000 [06:27<13:14,  1.18s/it]
Epoch Progress:  33%|███████▊                | 328/1000 [06:28<13:12,  1.18s/it]
Epoch Progress:  33%|███████▉                | 329/1000 [06:29<13:10,  1.18s/it]
Epoch Progress:  33%|███████▉                | 330/1000 [06:30<13:08,  1.18s/it]Dataset Size: 1%, Epoch: 331, Train Loss: 9.375272631645203, Val Loss: 10.874463923565752

Epoch Progress:  33%|███████▉                | 331/1000 [06:31<13:08,  1.18s/it]
Epoch Progress:  33%|███████▉                | 332/1000 [06:33<13:06,  1.18s/it]
Epoch Progress:  33%|███████▉                | 333/1000 [06:34<13:06,  1.18s/it]
Epoch Progress:  33%|████████                | 334/1000 [06:35<13:04,  1.18s/it]
Epoch Progress:  34%|████████                | 335/1000 [06:36<13:02,  1.18s/it]
Epoch Progress:  34%|████████                | 336/1000 [06:37<13:00,  1.18s/it]
Epoch Progress:  34%|████████                | 337/1000 [06:38<12:59,  1.18s/it]
Epoch Progress:  34%|████████                | 338/1000 [06:40<12:58,  1.18s/it]
Epoch Progress:  34%|████████▏               | 339/1000 [06:41<12:58,  1.18s/it]
Epoch Progress:  34%|████████▏               | 340/1000 [06:42<12:59,  1.18s/it]Dataset Size: 1%, Epoch: 341, Train Loss: 9.363399863243103, Val Loss: 10.888842334995022

Epoch Progress:  34%|████████▏               | 341/1000 [06:43<12:57,  1.18s/it]
Epoch Progress:  34%|████████▏               | 342/1000 [06:44<12:55,  1.18s/it]
Epoch Progress:  34%|████████▏               | 343/1000 [06:45<12:53,  1.18s/it]
Epoch Progress:  34%|████████▎               | 344/1000 [06:47<12:57,  1.19s/it]
Epoch Progress:  34%|████████▎               | 345/1000 [06:48<12:56,  1.19s/it]
Epoch Progress:  35%|████████▎               | 346/1000 [06:49<12:53,  1.18s/it]
Epoch Progress:  35%|████████▎               | 347/1000 [06:50<12:51,  1.18s/it]
Epoch Progress:  35%|████████▎               | 348/1000 [06:51<12:52,  1.18s/it]
Epoch Progress:  35%|████████▍               | 349/1000 [06:53<12:48,  1.18s/it]
Epoch Progress:  35%|████████▍               | 350/1000 [06:54<12:46,  1.18s/it]Dataset Size: 1%, Epoch: 351, Train Loss: 9.36107623577118, Val Loss: 10.897081783839635

Epoch Progress:  35%|████████▍               | 351/1000 [06:55<12:44,  1.18s/it]
Epoch Progress:  35%|████████▍               | 352/1000 [06:56<12:43,  1.18s/it]
Epoch Progress:  35%|████████▍               | 353/1000 [06:57<12:42,  1.18s/it]
Epoch Progress:  35%|████████▍               | 354/1000 [06:58<12:41,  1.18s/it]
Epoch Progress:  36%|████████▌               | 355/1000 [07:00<12:39,  1.18s/it]
Epoch Progress:  36%|████████▌               | 356/1000 [07:01<12:39,  1.18s/it]
Epoch Progress:  36%|████████▌               | 357/1000 [07:02<12:38,  1.18s/it]
Epoch Progress:  36%|████████▌               | 358/1000 [07:03<12:36,  1.18s/it]
Epoch Progress:  36%|████████▌               | 359/1000 [07:04<12:35,  1.18s/it]
Epoch Progress:  36%|████████▋               | 360/1000 [07:06<12:34,  1.18s/it]Dataset Size: 1%, Epoch: 361, Train Loss: 9.385272026062012, Val Loss: 10.906944596922242

Epoch Progress:  36%|████████▋               | 361/1000 [07:07<12:32,  1.18s/it]
Epoch Progress:  36%|████████▋               | 362/1000 [07:08<12:30,  1.18s/it]
Epoch Progress:  36%|████████▋               | 363/1000 [07:09<12:29,  1.18s/it]
Epoch Progress:  36%|████████▋               | 364/1000 [07:10<12:29,  1.18s/it]
Epoch Progress:  36%|████████▊               | 365/1000 [07:11<12:30,  1.18s/it]
Epoch Progress:  37%|████████▊               | 366/1000 [07:13<12:28,  1.18s/it]
Epoch Progress:  37%|████████▊               | 367/1000 [07:14<12:26,  1.18s/it]
Epoch Progress:  37%|████████▊               | 368/1000 [07:15<12:25,  1.18s/it]
Epoch Progress:  37%|████████▊               | 369/1000 [07:16<12:23,  1.18s/it]
Epoch Progress:  37%|████████▉               | 370/1000 [07:17<12:23,  1.18s/it]Dataset Size: 1%, Epoch: 371, Train Loss: 9.326666831970215, Val Loss: 10.916385762103193

Epoch Progress:  37%|████████▉               | 371/1000 [07:19<12:27,  1.19s/it]
Epoch Progress:  37%|████████▉               | 372/1000 [07:20<12:24,  1.19s/it]
Epoch Progress:  37%|████████▉               | 373/1000 [07:21<12:21,  1.18s/it]
Epoch Progress:  37%|████████▉               | 374/1000 [07:22<12:21,  1.18s/it]
Epoch Progress:  38%|█████████               | 375/1000 [07:23<12:19,  1.18s/it]
Epoch Progress:  38%|█████████               | 376/1000 [07:24<12:17,  1.18s/it]
Epoch Progress:  38%|█████████               | 377/1000 [07:26<12:16,  1.18s/it]
Epoch Progress:  38%|█████████               | 378/1000 [07:27<12:14,  1.18s/it]
Epoch Progress:  38%|█████████               | 379/1000 [07:28<12:13,  1.18s/it]
Epoch Progress:  38%|█████████               | 380/1000 [07:29<12:12,  1.18s/it]Dataset Size: 1%, Epoch: 381, Train Loss: 9.2988121509552, Val Loss: 10.923755323732054

Epoch Progress:  38%|█████████▏              | 381/1000 [07:30<12:10,  1.18s/it]
Epoch Progress:  38%|█████████▏              | 382/1000 [07:32<12:10,  1.18s/it]
Epoch Progress:  38%|█████████▏              | 383/1000 [07:33<12:09,  1.18s/it]
Epoch Progress:  38%|█████████▏              | 384/1000 [07:34<12:09,  1.18s/it]
Epoch Progress:  38%|█████████▏              | 385/1000 [07:35<12:06,  1.18s/it]
Epoch Progress:  39%|█████████▎              | 386/1000 [07:36<12:06,  1.18s/it]
Epoch Progress:  39%|█████████▎              | 387/1000 [07:37<12:03,  1.18s/it]
Epoch Progress:  39%|█████████▎              | 388/1000 [07:39<12:02,  1.18s/it]
Epoch Progress:  39%|█████████▎              | 389/1000 [07:40<12:00,  1.18s/it]
Epoch Progress:  39%|█████████▎              | 390/1000 [07:41<11:58,  1.18s/it]Dataset Size: 1%, Epoch: 391, Train Loss: 9.323002338409424, Val Loss: 10.933696932606884

Epoch Progress:  39%|█████████▍              | 391/1000 [07:42<11:58,  1.18s/it]
Epoch Progress:  39%|█████████▍              | 392/1000 [07:43<12:01,  1.19s/it]
Epoch Progress:  39%|█████████▍              | 393/1000 [07:45<11:59,  1.18s/it]
Epoch Progress:  39%|█████████▍              | 394/1000 [07:46<11:56,  1.18s/it]
Epoch Progress:  40%|█████████▍              | 395/1000 [07:47<11:55,  1.18s/it]
Epoch Progress:  40%|█████████▌              | 396/1000 [07:48<11:59,  1.19s/it]
Epoch Progress:  40%|█████████▌              | 397/1000 [07:49<12:07,  1.21s/it]
Epoch Progress:  40%|█████████▌              | 398/1000 [07:51<12:01,  1.20s/it]
Epoch Progress:  40%|█████████▌              | 399/1000 [07:52<11:59,  1.20s/it]
Epoch Progress:  40%|█████████▌              | 400/1000 [07:53<11:56,  1.19s/it]Dataset Size: 1%, Epoch: 401, Train Loss: 9.29511022567749, Val Loss: 10.939471975549475

Epoch Progress:  40%|█████████▌              | 401/1000 [07:54<11:52,  1.19s/it]
Epoch Progress:  40%|█████████▋              | 402/1000 [07:55<11:50,  1.19s/it]
Epoch Progress:  40%|█████████▋              | 403/1000 [07:56<11:47,  1.19s/it]
Epoch Progress:  40%|█████████▋              | 404/1000 [07:58<11:46,  1.19s/it]
Epoch Progress:  40%|█████████▋              | 405/1000 [07:59<11:45,  1.19s/it]
Epoch Progress:  41%|█████████▋              | 406/1000 [08:00<11:43,  1.18s/it]
Epoch Progress:  41%|█████████▊              | 407/1000 [08:01<11:55,  1.21s/it]
Epoch Progress:  41%|█████████▊              | 408/1000 [08:02<11:52,  1.20s/it]
Epoch Progress:  41%|█████████▊              | 409/1000 [08:04<11:46,  1.20s/it]
Epoch Progress:  41%|█████████▊              | 410/1000 [08:05<11:42,  1.19s/it]Dataset Size: 1%, Epoch: 411, Train Loss: 9.274935841560364, Val Loss: 10.945365695210246

Epoch Progress:  41%|█████████▊              | 411/1000 [08:06<11:39,  1.19s/it]
Epoch Progress:  41%|█████████▉              | 412/1000 [08:07<11:36,  1.18s/it]
Epoch Progress:  41%|█████████▉              | 413/1000 [08:08<11:35,  1.18s/it]
Epoch Progress:  41%|█████████▉              | 414/1000 [08:10<11:34,  1.18s/it]
Epoch Progress:  42%|█████████▉              | 415/1000 [08:11<11:33,  1.19s/it]
Epoch Progress:  42%|█████████▉              | 416/1000 [08:12<11:34,  1.19s/it]
Epoch Progress:  42%|██████████              | 417/1000 [08:13<11:31,  1.19s/it]
Epoch Progress:  42%|██████████              | 418/1000 [08:14<11:29,  1.18s/it]
Epoch Progress:  42%|██████████              | 419/1000 [08:15<11:27,  1.18s/it]
Epoch Progress:  42%|██████████              | 420/1000 [08:17<11:25,  1.18s/it]Dataset Size: 1%, Epoch: 421, Train Loss: 9.289551973342896, Val Loss: 10.94794815856141

Epoch Progress:  42%|██████████              | 421/1000 [08:18<11:24,  1.18s/it]
Epoch Progress:  42%|██████████▏             | 422/1000 [08:19<11:22,  1.18s/it]
Epoch Progress:  42%|██████████▏             | 423/1000 [08:20<11:19,  1.18s/it]
Epoch Progress:  42%|██████████▏             | 424/1000 [08:21<11:25,  1.19s/it]
Epoch Progress:  42%|██████████▏             | 425/1000 [08:23<11:22,  1.19s/it]
Epoch Progress:  43%|██████████▏             | 426/1000 [08:24<11:19,  1.18s/it]
Epoch Progress:  43%|██████████▏             | 427/1000 [08:25<11:16,  1.18s/it]
Epoch Progress:  43%|██████████▎             | 428/1000 [08:26<11:15,  1.18s/it]
Epoch Progress:  43%|██████████▎             | 429/1000 [08:27<11:13,  1.18s/it]
Epoch Progress:  43%|██████████▎             | 430/1000 [08:28<11:12,  1.18s/it]Dataset Size: 1%, Epoch: 431, Train Loss: 9.275710344314575, Val Loss: 10.955534191874715

Epoch Progress:  43%|██████████▎             | 431/1000 [08:30<11:11,  1.18s/it]
Epoch Progress:  43%|██████████▎             | 432/1000 [08:31<11:10,  1.18s/it]
Epoch Progress:  43%|██████████▍             | 433/1000 [08:32<11:09,  1.18s/it]
Epoch Progress:  43%|██████████▍             | 434/1000 [08:33<11:07,  1.18s/it]
Epoch Progress:  44%|██████████▍             | 435/1000 [08:34<11:06,  1.18s/it]
Epoch Progress:  44%|██████████▍             | 436/1000 [08:36<11:05,  1.18s/it]
Epoch Progress:  44%|██████████▍             | 437/1000 [08:37<11:03,  1.18s/it]
Epoch Progress:  44%|██████████▌             | 438/1000 [08:38<11:08,  1.19s/it]
Epoch Progress:  44%|██████████▌             | 439/1000 [08:39<11:05,  1.19s/it]
Epoch Progress:  44%|██████████▌             | 440/1000 [08:40<11:03,  1.18s/it]Dataset Size: 1%, Epoch: 441, Train Loss: 9.297302722930908, Val Loss: 10.96147194156399

Epoch Progress:  44%|██████████▌             | 441/1000 [08:41<11:02,  1.19s/it]
Epoch Progress:  44%|██████████▌             | 442/1000 [08:43<11:01,  1.18s/it]
Epoch Progress:  44%|██████████▋             | 443/1000 [08:44<10:59,  1.18s/it]
Epoch Progress:  44%|██████████▋             | 444/1000 [08:45<10:57,  1.18s/it]
Epoch Progress:  44%|██████████▋             | 445/1000 [08:46<10:55,  1.18s/it]
Epoch Progress:  45%|██████████▋             | 446/1000 [08:47<10:54,  1.18s/it]
Epoch Progress:  45%|██████████▋             | 447/1000 [08:49<10:52,  1.18s/it]
Epoch Progress:  45%|██████████▊             | 448/1000 [08:50<10:51,  1.18s/it]
Epoch Progress:  45%|██████████▊             | 449/1000 [08:51<10:50,  1.18s/it]
Epoch Progress:  45%|██████████▊             | 450/1000 [08:52<10:54,  1.19s/it]Dataset Size: 1%, Epoch: 451, Train Loss: 9.2423677444458, Val Loss: 10.965327225722275

Epoch Progress:  45%|██████████▊             | 451/1000 [08:53<10:52,  1.19s/it]
Epoch Progress:  45%|██████████▊             | 452/1000 [08:55<10:51,  1.19s/it]
Epoch Progress:  45%|██████████▊             | 453/1000 [08:56<10:49,  1.19s/it]
Epoch Progress:  45%|██████████▉             | 454/1000 [08:57<10:46,  1.18s/it]
Epoch Progress:  46%|██████████▉             | 455/1000 [08:58<10:44,  1.18s/it]
Epoch Progress:  46%|██████████▉             | 456/1000 [08:59<10:43,  1.18s/it]
Epoch Progress:  46%|██████████▉             | 457/1000 [09:00<10:43,  1.19s/it]
Epoch Progress:  46%|██████████▉             | 458/1000 [09:02<10:44,  1.19s/it]
Epoch Progress:  46%|███████████             | 459/1000 [09:03<10:41,  1.19s/it]
Epoch Progress:  46%|███████████             | 460/1000 [09:04<10:39,  1.18s/it]Dataset Size: 1%, Epoch: 461, Train Loss: 9.250150084495544, Val Loss: 10.96759420865542

Epoch Progress:  46%|███████████             | 461/1000 [09:05<10:37,  1.18s/it]
Epoch Progress:  46%|███████████             | 462/1000 [09:06<10:35,  1.18s/it]
Epoch Progress:  46%|███████████             | 463/1000 [09:08<10:35,  1.18s/it]
Epoch Progress:  46%|███████████▏            | 464/1000 [09:09<10:32,  1.18s/it]
Epoch Progress:  46%|███████████▏            | 465/1000 [09:10<10:31,  1.18s/it]
Epoch Progress:  47%|███████████▏            | 466/1000 [09:11<10:32,  1.18s/it]
Epoch Progress:  47%|███████████▏            | 467/1000 [09:12<10:29,  1.18s/it]
Epoch Progress:  47%|███████████▏            | 468/1000 [09:13<10:27,  1.18s/it]
Epoch Progress:  47%|███████████▎            | 469/1000 [09:15<10:26,  1.18s/it]
Epoch Progress:  47%|███████████▎            | 470/1000 [09:16<10:26,  1.18s/it]Dataset Size: 1%, Epoch: 471, Train Loss: 9.247560262680054, Val Loss: 10.975573168172465

Epoch Progress:  47%|███████████▎            | 471/1000 [09:17<10:24,  1.18s/it]
Epoch Progress:  47%|███████████▎            | 472/1000 [09:18<10:24,  1.18s/it]
Epoch Progress:  47%|███████████▎            | 473/1000 [09:19<10:23,  1.18s/it]
Epoch Progress:  47%|███████████▍            | 474/1000 [09:21<10:21,  1.18s/it]
Epoch Progress:  48%|███████████▍            | 475/1000 [09:22<10:21,  1.18s/it]
Epoch Progress:  48%|███████████▍            | 476/1000 [09:23<10:18,  1.18s/it]
Epoch Progress:  48%|███████████▍            | 477/1000 [09:24<10:20,  1.19s/it]
Epoch Progress:  48%|███████████▍            | 478/1000 [09:25<10:17,  1.18s/it]
Epoch Progress:  48%|███████████▍            | 479/1000 [09:26<10:15,  1.18s/it]
Epoch Progress:  48%|███████████▌            | 480/1000 [09:28<10:12,  1.18s/it]Dataset Size: 1%, Epoch: 481, Train Loss: 9.216339707374573, Val Loss: 10.979835832273805

Epoch Progress:  48%|███████████▌            | 481/1000 [09:29<10:11,  1.18s/it]
Epoch Progress:  48%|███████████▌            | 482/1000 [09:30<10:10,  1.18s/it]
Epoch Progress:  48%|███████████▌            | 483/1000 [09:31<10:10,  1.18s/it]
Epoch Progress:  48%|███████████▌            | 484/1000 [09:32<10:09,  1.18s/it]
Epoch Progress:  48%|███████████▋            | 485/1000 [09:34<10:07,  1.18s/it]
Epoch Progress:  49%|███████████▋            | 486/1000 [09:35<10:06,  1.18s/it]
Epoch Progress:  49%|███████████▋            | 487/1000 [09:36<10:04,  1.18s/it]
Epoch Progress:  49%|███████████▋            | 488/1000 [09:37<10:02,  1.18s/it]
Epoch Progress:  49%|███████████▋            | 489/1000 [09:38<10:01,  1.18s/it]
Epoch Progress:  49%|███████████▊            | 490/1000 [09:39<10:00,  1.18s/it]Dataset Size: 1%, Epoch: 491, Train Loss: 9.22383987903595, Val Loss: 10.983782334761186

Epoch Progress:  49%|███████████▊            | 491/1000 [09:41<09:59,  1.18s/it]
Epoch Progress:  49%|███████████▊            | 492/1000 [09:42<10:00,  1.18s/it]
Epoch Progress:  49%|███████████▊            | 493/1000 [09:43<09:58,  1.18s/it]
Epoch Progress:  49%|███████████▊            | 494/1000 [09:44<09:56,  1.18s/it]
Epoch Progress:  50%|███████████▉            | 495/1000 [09:45<09:55,  1.18s/it]
Epoch Progress:  50%|███████████▉            | 496/1000 [09:46<09:54,  1.18s/it]
Epoch Progress:  50%|███████████▉            | 497/1000 [09:48<09:53,  1.18s/it]
Epoch Progress:  50%|███████████▉            | 498/1000 [09:49<09:52,  1.18s/it]
Epoch Progress:  50%|███████████▉            | 499/1000 [09:50<09:51,  1.18s/it]
Epoch Progress:  50%|████████████            | 500/1000 [09:51<09:52,  1.18s/it]Dataset Size: 1%, Epoch: 501, Train Loss: 9.230677247047424, Val Loss: 10.982293178508808

Epoch Progress:  50%|████████████            | 501/1000 [09:52<09:50,  1.18s/it]
Epoch Progress:  50%|████████████            | 502/1000 [09:54<09:48,  1.18s/it]
Epoch Progress:  50%|████████████            | 503/1000 [09:55<09:50,  1.19s/it]
Epoch Progress:  50%|████████████            | 504/1000 [09:56<09:49,  1.19s/it]
Epoch Progress:  50%|████████████            | 505/1000 [09:57<09:46,  1.19s/it]
Epoch Progress:  51%|████████████▏           | 506/1000 [09:58<09:44,  1.18s/it]
Epoch Progress:  51%|████████████▏           | 507/1000 [09:59<09:41,  1.18s/it]
Epoch Progress:  51%|████████████▏           | 508/1000 [10:01<09:40,  1.18s/it]
Epoch Progress:  51%|████████████▏           | 509/1000 [10:02<09:40,  1.18s/it]
Epoch Progress:  51%|████████████▏           | 510/1000 [10:03<09:38,  1.18s/it]Dataset Size: 1%, Epoch: 511, Train Loss: 9.224906921386719, Val Loss: 10.984573339487051

Epoch Progress:  51%|████████████▎           | 511/1000 [10:04<09:36,  1.18s/it]
Epoch Progress:  51%|████████████▎           | 512/1000 [10:05<09:35,  1.18s/it]
Epoch Progress:  51%|████████████▎           | 513/1000 [10:07<09:34,  1.18s/it]
Epoch Progress:  51%|████████████▎           | 514/1000 [10:08<09:33,  1.18s/it]
Epoch Progress:  52%|████████████▎           | 515/1000 [10:09<09:31,  1.18s/it]
Epoch Progress:  52%|████████████▍           | 516/1000 [10:10<09:30,  1.18s/it]
Epoch Progress:  52%|████████████▍           | 517/1000 [10:11<09:31,  1.18s/it]
Epoch Progress:  52%|████████████▍           | 518/1000 [10:12<09:29,  1.18s/it]
Epoch Progress:  52%|████████████▍           | 519/1000 [10:14<09:27,  1.18s/it]
Epoch Progress:  52%|████████████▍           | 520/1000 [10:15<09:25,  1.18s/it]Dataset Size: 1%, Epoch: 521, Train Loss: 9.219847083091736, Val Loss: 10.986405496473436

Epoch Progress:  52%|████████████▌           | 521/1000 [10:16<09:24,  1.18s/it]
Epoch Progress:  52%|████████████▌           | 522/1000 [10:17<09:23,  1.18s/it]
Epoch Progress:  52%|████████████▌           | 523/1000 [10:18<09:22,  1.18s/it]
Epoch Progress:  52%|████████████▌           | 524/1000 [10:20<09:21,  1.18s/it]
Epoch Progress:  52%|████████████▌           | 525/1000 [10:21<09:19,  1.18s/it]
Epoch Progress:  53%|████████████▌           | 526/1000 [10:22<09:19,  1.18s/it]
Epoch Progress:  53%|████████████▋           | 527/1000 [10:23<09:17,  1.18s/it]
Epoch Progress:  53%|████████████▋           | 528/1000 [10:24<09:16,  1.18s/it]
Epoch Progress:  53%|████████████▋           | 529/1000 [10:25<09:15,  1.18s/it]
Epoch Progress:  53%|████████████▋           | 530/1000 [10:27<09:18,  1.19s/it]Dataset Size: 1%, Epoch: 531, Train Loss: 9.205726623535156, Val Loss: 10.990298494116052

Epoch Progress:  53%|████████████▋           | 531/1000 [10:28<09:16,  1.19s/it]
Epoch Progress:  53%|████████████▊           | 532/1000 [10:29<09:13,  1.18s/it]
Epoch Progress:  53%|████████████▊           | 533/1000 [10:30<09:12,  1.18s/it]
Epoch Progress:  53%|████████████▊           | 534/1000 [10:31<09:11,  1.18s/it]
Epoch Progress:  54%|████████████▊           | 535/1000 [10:33<09:09,  1.18s/it]
Epoch Progress:  54%|████████████▊           | 536/1000 [10:34<09:07,  1.18s/it]
Epoch Progress:  54%|████████████▉           | 537/1000 [10:35<09:06,  1.18s/it]
Epoch Progress:  54%|████████████▉           | 538/1000 [10:36<09:05,  1.18s/it]
Epoch Progress:  54%|████████████▉           | 539/1000 [10:37<09:04,  1.18s/it]
Epoch Progress:  54%|████████████▉           | 540/1000 [10:38<09:02,  1.18s/it]Dataset Size: 1%, Epoch: 541, Train Loss: 9.198970437049866, Val Loss: 10.990123897403866

Epoch Progress:  54%|████████████▉           | 541/1000 [10:40<09:00,  1.18s/it]
Epoch Progress:  54%|█████████████           | 542/1000 [10:41<08:59,  1.18s/it]
Epoch Progress:  54%|█████████████           | 543/1000 [10:42<08:59,  1.18s/it]
Epoch Progress:  54%|█████████████           | 544/1000 [10:43<08:57,  1.18s/it]
Epoch Progress:  55%|█████████████           | 545/1000 [10:44<08:56,  1.18s/it]
Epoch Progress:  55%|█████████████           | 546/1000 [10:46<08:54,  1.18s/it]
Epoch Progress:  55%|█████████████▏          | 547/1000 [10:47<08:53,  1.18s/it]
Epoch Progress:  55%|█████████████▏          | 548/1000 [10:48<08:52,  1.18s/it]
Epoch Progress:  55%|█████████████▏          | 549/1000 [10:49<08:51,  1.18s/it]
Epoch Progress:  55%|█████████████▏          | 550/1000 [10:50<08:50,  1.18s/it]Dataset Size: 1%, Epoch: 551, Train Loss: 9.197029709815979, Val Loss: 10.9914365000539

Epoch Progress:  55%|█████████████▏          | 551/1000 [10:51<08:50,  1.18s/it]
Epoch Progress:  55%|█████████████▏          | 552/1000 [10:53<08:48,  1.18s/it]
Epoch Progress:  55%|█████████████▎          | 553/1000 [10:54<08:47,  1.18s/it]
Epoch Progress:  55%|█████████████▎          | 554/1000 [10:55<08:46,  1.18s/it]
Epoch Progress:  56%|█████████████▎          | 555/1000 [10:56<08:45,  1.18s/it]
Epoch Progress:  56%|█████████████▎          | 556/1000 [10:57<08:47,  1.19s/it]
Epoch Progress:  56%|█████████████▎          | 557/1000 [10:59<08:46,  1.19s/it]
Epoch Progress:  56%|█████████████▍          | 558/1000 [11:00<08:44,  1.19s/it]
Epoch Progress:  56%|█████████████▍          | 559/1000 [11:01<08:42,  1.18s/it]
Epoch Progress:  56%|█████████████▍          | 560/1000 [11:02<08:42,  1.19s/it]Dataset Size: 1%, Epoch: 561, Train Loss: 9.223848342895508, Val Loss: 10.993176868983678

Epoch Progress:  56%|█████████████▍          | 561/1000 [11:03<08:40,  1.18s/it]
Epoch Progress:  56%|█████████████▍          | 562/1000 [11:04<08:38,  1.18s/it]
Epoch Progress:  56%|█████████████▌          | 563/1000 [11:06<08:36,  1.18s/it]
Epoch Progress:  56%|█████████████▌          | 564/1000 [11:07<08:34,  1.18s/it]
Epoch Progress:  56%|█████████████▌          | 565/1000 [11:08<08:32,  1.18s/it]
Epoch Progress:  57%|█████████████▌          | 566/1000 [11:09<08:31,  1.18s/it]
Epoch Progress:  57%|█████████████▌          | 567/1000 [11:10<08:29,  1.18s/it]
Epoch Progress:  57%|█████████████▋          | 568/1000 [11:12<08:30,  1.18s/it]
Epoch Progress:  57%|█████████████▋          | 569/1000 [11:13<08:28,  1.18s/it]
Epoch Progress:  57%|█████████████▋          | 570/1000 [11:14<08:27,  1.18s/it]Dataset Size: 1%, Epoch: 571, Train Loss: 9.19044280052185, Val Loss: 10.993995010078727

Epoch Progress:  57%|█████████████▋          | 571/1000 [11:15<08:27,  1.18s/it]
Epoch Progress:  57%|█████████████▋          | 572/1000 [11:16<08:28,  1.19s/it]
Epoch Progress:  57%|█████████████▊          | 573/1000 [11:17<08:29,  1.19s/it]
Epoch Progress:  57%|█████████████▊          | 574/1000 [11:19<08:27,  1.19s/it]
Epoch Progress:  57%|█████████████▊          | 575/1000 [11:20<08:25,  1.19s/it]
Epoch Progress:  58%|█████████████▊          | 576/1000 [11:21<08:24,  1.19s/it]
Epoch Progress:  58%|█████████████▊          | 577/1000 [11:22<08:22,  1.19s/it]
Epoch Progress:  58%|█████████████▊          | 578/1000 [11:23<08:20,  1.19s/it]
Epoch Progress:  58%|█████████████▉          | 579/1000 [11:25<08:18,  1.18s/it]
Epoch Progress:  58%|█████████████▉          | 580/1000 [11:26<08:17,  1.18s/it]Dataset Size: 1%, Epoch: 581, Train Loss: 9.183631539344788, Val Loss: 10.997848250649191

Epoch Progress:  58%|█████████████▉          | 581/1000 [11:27<08:15,  1.18s/it]
Epoch Progress:  58%|█████████████▉          | 582/1000 [11:28<08:14,  1.18s/it]
Epoch Progress:  58%|█████████████▉          | 583/1000 [11:29<08:17,  1.19s/it]
Epoch Progress:  58%|██████████████          | 584/1000 [11:31<08:15,  1.19s/it]
Epoch Progress:  58%|██████████████          | 585/1000 [11:32<08:34,  1.24s/it]
Epoch Progress:  59%|██████████████          | 586/1000 [11:33<08:26,  1.22s/it]
Epoch Progress:  59%|██████████████          | 587/1000 [11:34<08:19,  1.21s/it]
Epoch Progress:  59%|██████████████          | 588/1000 [11:35<08:14,  1.20s/it]
Epoch Progress:  59%|██████████████▏         | 589/1000 [11:37<08:10,  1.19s/it]
Epoch Progress:  59%|██████████████▏         | 590/1000 [11:38<08:09,  1.19s/it]Dataset Size: 1%, Epoch: 591, Train Loss: 9.196494221687317, Val Loss: 10.997458383634493

Epoch Progress:  59%|██████████████▏         | 591/1000 [11:39<08:06,  1.19s/it]
Epoch Progress:  59%|██████████████▏         | 592/1000 [11:40<08:03,  1.19s/it]
Epoch Progress:  59%|██████████████▏         | 593/1000 [11:41<08:03,  1.19s/it]
Epoch Progress:  59%|██████████████▎         | 594/1000 [11:43<08:02,  1.19s/it]
Epoch Progress:  60%|██████████████▎         | 595/1000 [11:44<08:00,  1.19s/it]
Epoch Progress:  60%|██████████████▎         | 596/1000 [11:45<07:59,  1.19s/it]
Epoch Progress:  60%|██████████████▎         | 597/1000 [11:46<07:57,  1.19s/it]
Epoch Progress:  60%|██████████████▎         | 598/1000 [11:47<07:56,  1.19s/it]
Epoch Progress:  60%|██████████████▍         | 599/1000 [11:48<07:56,  1.19s/it]
Epoch Progress:  60%|██████████████▍         | 600/1000 [11:50<07:54,  1.19s/it]Dataset Size: 1%, Epoch: 601, Train Loss: 9.1558678150177, Val Loss: 11.001299065428896

Epoch Progress:  60%|██████████████▍         | 601/1000 [11:51<07:53,  1.19s/it]
Epoch Progress:  60%|██████████████▍         | 602/1000 [11:52<07:51,  1.19s/it]
Epoch Progress:  60%|██████████████▍         | 603/1000 [11:53<07:49,  1.18s/it]
Epoch Progress:  60%|██████████████▍         | 604/1000 [11:54<07:48,  1.18s/it]
Epoch Progress:  60%|██████████████▌         | 605/1000 [11:56<07:46,  1.18s/it]
Epoch Progress:  61%|██████████████▌         | 606/1000 [11:57<07:45,  1.18s/it]
Epoch Progress:  61%|██████████████▌         | 607/1000 [11:58<07:45,  1.18s/it]
Epoch Progress:  61%|██████████████▌         | 608/1000 [11:59<07:43,  1.18s/it]
Epoch Progress:  61%|██████████████▌         | 609/1000 [12:00<07:46,  1.19s/it]
Epoch Progress:  61%|██████████████▋         | 610/1000 [12:02<07:45,  1.19s/it]Dataset Size: 1%, Epoch: 611, Train Loss: 9.19361937046051, Val Loss: 10.997176740076634

Epoch Progress:  61%|██████████████▋         | 611/1000 [12:03<07:43,  1.19s/it]
Epoch Progress:  61%|██████████████▋         | 612/1000 [12:04<07:40,  1.19s/it]
Epoch Progress:  61%|██████████████▋         | 613/1000 [12:05<07:37,  1.18s/it]
Epoch Progress:  61%|██████████████▋         | 614/1000 [12:06<07:35,  1.18s/it]
Epoch Progress:  62%|██████████████▊         | 615/1000 [12:07<07:34,  1.18s/it]
Epoch Progress:  62%|██████████████▊         | 616/1000 [12:09<07:33,  1.18s/it]
Epoch Progress:  62%|██████████████▊         | 617/1000 [12:10<07:31,  1.18s/it]
Epoch Progress:  62%|██████████████▊         | 618/1000 [12:11<07:31,  1.18s/it]
Epoch Progress:  62%|██████████████▊         | 619/1000 [12:12<07:30,  1.18s/it]
Epoch Progress:  62%|██████████████▉         | 620/1000 [12:13<07:28,  1.18s/it]Dataset Size: 1%, Epoch: 621, Train Loss: 9.145246267318726, Val Loss: 11.000010849593522

Epoch Progress:  62%|██████████████▉         | 621/1000 [12:14<07:26,  1.18s/it]
Epoch Progress:  62%|██████████████▉         | 622/1000 [12:16<07:26,  1.18s/it]
Epoch Progress:  62%|██████████████▉         | 623/1000 [12:17<07:26,  1.18s/it]
Epoch Progress:  62%|██████████████▉         | 624/1000 [12:18<07:24,  1.18s/it]
Epoch Progress:  62%|███████████████         | 625/1000 [12:19<07:23,  1.18s/it]
Epoch Progress:  63%|███████████████         | 626/1000 [12:20<07:22,  1.18s/it]
Epoch Progress:  63%|███████████████         | 627/1000 [12:22<07:22,  1.19s/it]
Epoch Progress:  63%|███████████████         | 628/1000 [12:23<07:20,  1.18s/it]
Epoch Progress:  63%|███████████████         | 629/1000 [12:24<07:18,  1.18s/it]
Epoch Progress:  63%|███████████████         | 630/1000 [12:25<07:16,  1.18s/it]Dataset Size: 1%, Epoch: 631, Train Loss: 9.161792993545532, Val Loss: 11.003505682016348

Epoch Progress:  63%|███████████████▏        | 631/1000 [12:26<07:15,  1.18s/it]
Epoch Progress:  63%|███████████████▏        | 632/1000 [12:28<07:14,  1.18s/it]
Epoch Progress:  63%|███████████████▏        | 633/1000 [12:29<07:12,  1.18s/it]
Epoch Progress:  63%|███████████████▏        | 634/1000 [12:30<07:11,  1.18s/it]
Epoch Progress:  64%|███████████████▏        | 635/1000 [12:31<07:11,  1.18s/it]
Epoch Progress:  64%|███████████████▎        | 636/1000 [12:32<07:12,  1.19s/it]
Epoch Progress:  64%|███████████████▎        | 637/1000 [12:33<07:10,  1.19s/it]
Epoch Progress:  64%|███████████████▎        | 638/1000 [12:35<07:08,  1.18s/it]
Epoch Progress:  64%|███████████████▎        | 639/1000 [12:36<07:06,  1.18s/it]
Epoch Progress:  64%|███████████████▎        | 640/1000 [12:37<07:05,  1.18s/it]Dataset Size: 1%, Epoch: 641, Train Loss: 9.150226950645447, Val Loss: 11.001932751048695

Epoch Progress:  64%|███████████████▍        | 641/1000 [12:38<07:04,  1.18s/it]
Epoch Progress:  64%|███████████████▍        | 642/1000 [12:39<07:02,  1.18s/it]
Epoch Progress:  64%|███████████████▍        | 643/1000 [12:41<07:00,  1.18s/it]
Epoch Progress:  64%|███████████████▍        | 644/1000 [12:42<07:01,  1.18s/it]
Epoch Progress:  64%|███████████████▍        | 645/1000 [12:43<06:59,  1.18s/it]
Epoch Progress:  65%|███████████████▌        | 646/1000 [12:44<06:57,  1.18s/it]
Epoch Progress:  65%|███████████████▌        | 647/1000 [12:45<06:56,  1.18s/it]
Epoch Progress:  65%|███████████████▌        | 648/1000 [12:46<06:54,  1.18s/it]
Epoch Progress:  65%|███████████████▌        | 649/1000 [12:48<06:54,  1.18s/it]
Epoch Progress:  65%|███████████████▌        | 650/1000 [12:49<06:54,  1.18s/it]Dataset Size: 1%, Epoch: 651, Train Loss: 9.155674576759338, Val Loss: 11.001623475706422

Epoch Progress:  65%|███████████████▌        | 651/1000 [12:50<06:53,  1.18s/it]
Epoch Progress:  65%|███████████████▋        | 652/1000 [12:51<06:55,  1.19s/it]
Epoch Progress:  65%|███████████████▋        | 653/1000 [12:52<06:54,  1.19s/it]
Epoch Progress:  65%|███████████████▋        | 654/1000 [12:54<06:52,  1.19s/it]
Epoch Progress:  66%|███████████████▋        | 655/1000 [12:55<06:50,  1.19s/it]
Epoch Progress:  66%|███████████████▋        | 656/1000 [12:56<06:48,  1.19s/it]
Epoch Progress:  66%|███████████████▊        | 657/1000 [12:57<06:47,  1.19s/it]
Epoch Progress:  66%|███████████████▊        | 658/1000 [12:58<06:48,  1.19s/it]
Epoch Progress:  66%|███████████████▊        | 659/1000 [13:00<06:47,  1.19s/it]
Epoch Progress:  66%|███████████████▊        | 660/1000 [13:01<06:44,  1.19s/it]Dataset Size: 1%, Epoch: 661, Train Loss: 9.186423897743225, Val Loss: 11.006204679414823

Epoch Progress:  66%|███████████████▊        | 661/1000 [13:02<06:43,  1.19s/it]
Epoch Progress:  66%|███████████████▉        | 662/1000 [13:03<06:45,  1.20s/it]
Epoch Progress:  66%|███████████████▉        | 663/1000 [13:04<06:43,  1.20s/it]
Epoch Progress:  66%|███████████████▉        | 664/1000 [13:06<06:41,  1.19s/it]
Epoch Progress:  66%|███████████████▉        | 665/1000 [13:07<06:39,  1.19s/it]
Epoch Progress:  67%|███████████████▉        | 666/1000 [13:08<06:37,  1.19s/it]
Epoch Progress:  67%|████████████████        | 667/1000 [13:09<06:35,  1.19s/it]
Epoch Progress:  67%|████████████████        | 668/1000 [13:10<06:34,  1.19s/it]
Epoch Progress:  67%|████████████████        | 669/1000 [13:11<06:34,  1.19s/it]
Epoch Progress:  67%|████████████████        | 670/1000 [13:13<06:33,  1.19s/it]Dataset Size: 1%, Epoch: 671, Train Loss: 9.161834955215454, Val Loss: 11.007511869653479

Epoch Progress:  67%|████████████████        | 671/1000 [13:14<06:31,  1.19s/it]
Epoch Progress:  67%|████████████████▏       | 672/1000 [13:15<06:29,  1.19s/it]
Epoch Progress:  67%|████████████████▏       | 673/1000 [13:16<06:27,  1.19s/it]
Epoch Progress:  67%|████████████████▏       | 674/1000 [13:17<06:25,  1.18s/it]
Epoch Progress:  68%|████████████████▏       | 675/1000 [13:19<06:24,  1.18s/it]
Epoch Progress:  68%|████████████████▏       | 676/1000 [13:20<06:23,  1.18s/it]
Epoch Progress:  68%|████████████████▏       | 677/1000 [13:21<06:22,  1.18s/it]
Epoch Progress:  68%|████████████████▎       | 678/1000 [13:22<06:21,  1.19s/it]
Epoch Progress:  68%|████████████████▎       | 679/1000 [13:23<06:20,  1.19s/it]
Epoch Progress:  68%|████████████████▎       | 680/1000 [13:24<06:18,  1.18s/it]Dataset Size: 1%, Epoch: 681, Train Loss: 9.15772557258606, Val Loss: 11.004964172066032

Epoch Progress:  68%|████████████████▎       | 681/1000 [13:26<06:17,  1.18s/it]
Epoch Progress:  68%|████████████████▎       | 682/1000 [13:27<06:15,  1.18s/it]
Epoch Progress:  68%|████████████████▍       | 683/1000 [13:28<06:14,  1.18s/it]
Epoch Progress:  68%|████████████████▍       | 684/1000 [13:29<06:12,  1.18s/it]
Epoch Progress:  68%|████████████████▍       | 685/1000 [13:30<06:11,  1.18s/it]
Epoch Progress:  69%|████████████████▍       | 686/1000 [13:32<06:11,  1.18s/it]
Epoch Progress:  69%|████████████████▍       | 687/1000 [13:33<06:09,  1.18s/it]
Epoch Progress:  69%|████████████████▌       | 688/1000 [13:34<06:09,  1.18s/it]
Epoch Progress:  69%|████████████████▌       | 689/1000 [13:35<06:08,  1.18s/it]
Epoch Progress:  69%|████████████████▌       | 690/1000 [13:36<06:06,  1.18s/it]Dataset Size: 1%, Epoch: 691, Train Loss: 9.149054527282715, Val Loss: 11.003870134229784

Epoch Progress:  69%|████████████████▌       | 691/1000 [13:37<06:05,  1.18s/it]
Epoch Progress:  69%|████████████████▌       | 692/1000 [13:39<06:03,  1.18s/it]
Epoch Progress:  69%|████████████████▋       | 693/1000 [13:40<06:02,  1.18s/it]
Epoch Progress:  69%|████████████████▋       | 694/1000 [13:41<06:01,  1.18s/it]
Epoch Progress:  70%|████████████████▋       | 695/1000 [13:42<06:00,  1.18s/it]
Epoch Progress:  70%|████████████████▋       | 696/1000 [13:43<05:58,  1.18s/it]
Epoch Progress:  70%|████████████████▋       | 697/1000 [13:45<05:57,  1.18s/it]
Epoch Progress:  70%|████████████████▊       | 698/1000 [13:46<05:55,  1.18s/it]
Epoch Progress:  70%|████████████████▊       | 699/1000 [13:47<05:54,  1.18s/it]
Epoch Progress:  70%|████████████████▊       | 700/1000 [13:48<05:53,  1.18s/it]Dataset Size: 1%, Epoch: 701, Train Loss: 9.181316018104553, Val Loss: 11.005048095405876

Epoch Progress:  70%|████████████████▊       | 701/1000 [13:49<05:51,  1.18s/it]
Epoch Progress:  70%|████████████████▊       | 702/1000 [13:50<05:50,  1.18s/it]
Epoch Progress:  70%|████████████████▊       | 703/1000 [13:52<05:51,  1.18s/it]
Epoch Progress:  70%|████████████████▉       | 704/1000 [13:53<05:49,  1.18s/it]
Epoch Progress:  70%|████████████████▉       | 705/1000 [13:54<05:48,  1.18s/it]
Epoch Progress:  71%|████████████████▉       | 706/1000 [13:55<05:46,  1.18s/it]
Epoch Progress:  71%|████████████████▉       | 707/1000 [13:56<05:45,  1.18s/it]
Epoch Progress:  71%|████████████████▉       | 708/1000 [13:58<05:44,  1.18s/it]
Epoch Progress:  71%|█████████████████       | 709/1000 [13:59<05:43,  1.18s/it]
Epoch Progress:  71%|█████████████████       | 710/1000 [14:00<05:42,  1.18s/it]Dataset Size: 1%, Epoch: 711, Train Loss: 9.164032578468323, Val Loss: 11.002885979491396

Epoch Progress:  71%|█████████████████       | 711/1000 [14:01<05:42,  1.18s/it]
Epoch Progress:  71%|█████████████████       | 712/1000 [14:02<05:41,  1.18s/it]
Epoch Progress:  71%|█████████████████       | 713/1000 [14:03<05:39,  1.18s/it]
Epoch Progress:  71%|█████████████████▏      | 714/1000 [14:05<05:38,  1.18s/it]
Epoch Progress:  72%|█████████████████▏      | 715/1000 [14:06<05:38,  1.19s/it]
Epoch Progress:  72%|█████████████████▏      | 716/1000 [14:07<05:36,  1.19s/it]
Epoch Progress:  72%|█████████████████▏      | 717/1000 [14:08<05:36,  1.19s/it]
Epoch Progress:  72%|█████████████████▏      | 718/1000 [14:09<05:34,  1.19s/it]
Epoch Progress:  72%|█████████████████▎      | 719/1000 [14:11<05:33,  1.19s/it]
Epoch Progress:  72%|█████████████████▎      | 720/1000 [14:12<05:33,  1.19s/it]Dataset Size: 1%, Epoch: 721, Train Loss: 9.173916697502136, Val Loss: 11.006509310239322

Epoch Progress:  72%|█████████████████▎      | 721/1000 [14:13<05:31,  1.19s/it]
Epoch Progress:  72%|█████████████████▎      | 722/1000 [14:14<05:30,  1.19s/it]
Epoch Progress:  72%|█████████████████▎      | 723/1000 [14:15<05:28,  1.19s/it]
Epoch Progress:  72%|█████████████████▍      | 724/1000 [14:16<05:27,  1.19s/it]
Epoch Progress:  72%|█████████████████▍      | 725/1000 [14:18<05:25,  1.18s/it]
Epoch Progress:  73%|█████████████████▍      | 726/1000 [14:19<05:24,  1.18s/it]
Epoch Progress:  73%|█████████████████▍      | 727/1000 [14:20<05:22,  1.18s/it]
Epoch Progress:  73%|█████████████████▍      | 728/1000 [14:21<05:21,  1.18s/it]
Epoch Progress:  73%|█████████████████▍      | 729/1000 [14:22<05:20,  1.18s/it]
Epoch Progress:  73%|█████████████████▌      | 730/1000 [14:24<05:18,  1.18s/it]Dataset Size: 1%, Epoch: 731, Train Loss: 9.132928967475891, Val Loss: 11.004315636374734

Epoch Progress:  73%|█████████████████▌      | 731/1000 [14:25<05:17,  1.18s/it]
Epoch Progress:  73%|█████████████████▌      | 732/1000 [14:26<05:15,  1.18s/it]
Epoch Progress:  73%|█████████████████▌      | 733/1000 [14:27<05:14,  1.18s/it]
Epoch Progress:  73%|█████████████████▌      | 734/1000 [14:28<05:13,  1.18s/it]
Epoch Progress:  74%|█████████████████▋      | 735/1000 [14:29<05:12,  1.18s/it]
Epoch Progress:  74%|█████████████████▋      | 736/1000 [14:31<05:11,  1.18s/it]
Epoch Progress:  74%|█████████████████▋      | 737/1000 [14:32<05:12,  1.19s/it]
Epoch Progress:  74%|█████████████████▋      | 738/1000 [14:33<05:11,  1.19s/it]
Epoch Progress:  74%|█████████████████▋      | 739/1000 [14:34<05:10,  1.19s/it]
Epoch Progress:  74%|█████████████████▊      | 740/1000 [14:35<05:09,  1.19s/it]Dataset Size: 1%, Epoch: 741, Train Loss: 9.152771592140198, Val Loss: 11.008736932432496

Epoch Progress:  74%|█████████████████▊      | 741/1000 [14:37<05:07,  1.19s/it]
Epoch Progress:  74%|█████████████████▊      | 742/1000 [14:38<05:06,  1.19s/it]
Epoch Progress:  74%|█████████████████▊      | 743/1000 [14:39<05:04,  1.19s/it]
Epoch Progress:  74%|█████████████████▊      | 744/1000 [14:40<05:02,  1.18s/it]
Epoch Progress:  74%|█████████████████▉      | 745/1000 [14:41<05:01,  1.18s/it]
Epoch Progress:  75%|█████████████████▉      | 746/1000 [14:43<05:00,  1.18s/it]
Epoch Progress:  75%|█████████████████▉      | 747/1000 [14:44<04:58,  1.18s/it]
Epoch Progress:  75%|█████████████████▉      | 748/1000 [14:45<04:57,  1.18s/it]
Epoch Progress:  75%|█████████████████▉      | 749/1000 [14:46<04:56,  1.18s/it]
Epoch Progress:  75%|██████████████████      | 750/1000 [14:47<04:54,  1.18s/it]Dataset Size: 1%, Epoch: 751, Train Loss: 9.13598620891571, Val Loss: 11.003951122234394

Epoch Progress:  75%|██████████████████      | 751/1000 [14:48<04:53,  1.18s/it]
Epoch Progress:  75%|██████████████████      | 752/1000 [14:50<04:52,  1.18s/it]
Epoch Progress:  75%|██████████████████      | 753/1000 [14:51<04:51,  1.18s/it]
Epoch Progress:  75%|██████████████████      | 754/1000 [14:52<04:51,  1.18s/it]
Epoch Progress:  76%|██████████████████      | 755/1000 [14:53<04:49,  1.18s/it]
Epoch Progress:  76%|██████████████████▏     | 756/1000 [14:54<04:48,  1.18s/it]
Epoch Progress:  76%|██████████████████▏     | 757/1000 [14:55<04:46,  1.18s/it]
Epoch Progress:  76%|██████████████████▏     | 758/1000 [14:57<04:45,  1.18s/it]
Epoch Progress:  76%|██████████████████▏     | 759/1000 [14:58<04:44,  1.18s/it]
Epoch Progress:  76%|██████████████████▏     | 760/1000 [14:59<04:42,  1.18s/it]Dataset Size: 1%, Epoch: 761, Train Loss: 9.153051137924194, Val Loss: 11.008645218688173

Epoch Progress:  76%|██████████████████▎     | 761/1000 [15:00<04:41,  1.18s/it]
Epoch Progress:  76%|██████████████████▎     | 762/1000 [15:01<04:40,  1.18s/it]
Epoch Progress:  76%|██████████████████▎     | 763/1000 [15:03<04:39,  1.18s/it]
Epoch Progress:  76%|██████████████████▎     | 764/1000 [15:04<04:38,  1.18s/it]
Epoch Progress:  76%|██████████████████▎     | 765/1000 [15:05<04:37,  1.18s/it]
Epoch Progress:  77%|██████████████████▍     | 766/1000 [15:06<04:35,  1.18s/it]
Epoch Progress:  77%|██████████████████▍     | 767/1000 [15:07<04:34,  1.18s/it]
Epoch Progress:  77%|██████████████████▍     | 768/1000 [15:08<04:34,  1.18s/it]
Epoch Progress:  77%|██████████████████▍     | 769/1000 [15:10<04:33,  1.18s/it]
Epoch Progress:  77%|██████████████████▍     | 770/1000 [15:11<04:31,  1.18s/it]Dataset Size: 1%, Epoch: 771, Train Loss: 9.140893816947937, Val Loss: 11.008310664783824

Epoch Progress:  77%|██████████████████▌     | 771/1000 [15:12<04:30,  1.18s/it]
Epoch Progress:  77%|██████████████████▌     | 772/1000 [15:13<04:29,  1.18s/it]
Epoch Progress:  77%|██████████████████▌     | 773/1000 [15:14<04:27,  1.18s/it]
Epoch Progress:  77%|██████████████████▌     | 774/1000 [15:16<04:26,  1.18s/it]
Epoch Progress:  78%|██████████████████▌     | 775/1000 [15:17<04:25,  1.18s/it]
Epoch Progress:  78%|██████████████████▌     | 776/1000 [15:18<04:24,  1.18s/it]
Epoch Progress:  78%|██████████████████▋     | 777/1000 [15:19<04:23,  1.18s/it]
Epoch Progress:  78%|██████████████████▋     | 778/1000 [15:20<04:21,  1.18s/it]
Epoch Progress:  78%|██████████████████▋     | 779/1000 [15:21<04:21,  1.18s/it]
Epoch Progress:  78%|██████████████████▋     | 780/1000 [15:23<04:19,  1.18s/it]Dataset Size: 1%, Epoch: 781, Train Loss: 9.13873279094696, Val Loss: 11.009991249480805

Epoch Progress:  78%|██████████████████▋     | 781/1000 [15:24<04:18,  1.18s/it]
Epoch Progress:  78%|██████████████████▊     | 782/1000 [15:25<04:17,  1.18s/it]
Epoch Progress:  78%|██████████████████▊     | 783/1000 [15:26<04:15,  1.18s/it]
Epoch Progress:  78%|██████████████████▊     | 784/1000 [15:27<04:14,  1.18s/it]
Epoch Progress:  78%|██████████████████▊     | 785/1000 [15:29<04:13,  1.18s/it]
Epoch Progress:  79%|██████████████████▊     | 786/1000 [15:30<04:12,  1.18s/it]
Epoch Progress:  79%|██████████████████▉     | 787/1000 [15:31<04:11,  1.18s/it]
Epoch Progress:  79%|██████████████████▉     | 788/1000 [15:32<04:10,  1.18s/it]
Epoch Progress:  79%|██████████████████▉     | 789/1000 [15:33<04:09,  1.18s/it]
Epoch Progress:  79%|██████████████████▉     | 790/1000 [15:34<04:07,  1.18s/it]Dataset Size: 1%, Epoch: 791, Train Loss: 9.11594271659851, Val Loss: 11.008585372528472

Epoch Progress:  79%|██████████████████▉     | 791/1000 [15:36<04:06,  1.18s/it]
Epoch Progress:  79%|███████████████████     | 792/1000 [15:37<04:04,  1.18s/it]
Epoch Progress:  79%|███████████████████     | 793/1000 [15:38<04:03,  1.18s/it]
Epoch Progress:  79%|███████████████████     | 794/1000 [15:39<04:02,  1.18s/it]
Epoch Progress:  80%|███████████████████     | 795/1000 [15:40<04:04,  1.19s/it]
Epoch Progress:  80%|███████████████████     | 796/1000 [15:42<04:03,  1.19s/it]
Epoch Progress:  80%|███████████████████▏    | 797/1000 [15:43<04:01,  1.19s/it]
Epoch Progress:  80%|███████████████████▏    | 798/1000 [15:44<03:59,  1.19s/it]
Epoch Progress:  80%|███████████████████▏    | 799/1000 [15:45<03:57,  1.18s/it]
Epoch Progress:  80%|███████████████████▏    | 800/1000 [15:46<03:56,  1.18s/it]Dataset Size: 1%, Epoch: 801, Train Loss: 9.188034415245056, Val Loss: 11.010120255606514

Epoch Progress:  80%|███████████████████▏    | 801/1000 [15:47<03:54,  1.18s/it]
Epoch Progress:  80%|███████████████████▏    | 802/1000 [15:49<03:53,  1.18s/it]
Epoch Progress:  80%|███████████████████▎    | 803/1000 [15:50<03:52,  1.18s/it]
Epoch Progress:  80%|███████████████████▎    | 804/1000 [15:51<03:51,  1.18s/it]
Epoch Progress:  80%|███████████████████▎    | 805/1000 [15:52<03:50,  1.18s/it]
Epoch Progress:  81%|███████████████████▎    | 806/1000 [15:53<03:48,  1.18s/it]
Epoch Progress:  81%|███████████████████▎    | 807/1000 [15:55<03:47,  1.18s/it]
Epoch Progress:  81%|███████████████████▍    | 808/1000 [15:56<03:46,  1.18s/it]
Epoch Progress:  81%|███████████████████▍    | 809/1000 [15:57<03:45,  1.18s/it]
Epoch Progress:  81%|███████████████████▍    | 810/1000 [15:58<03:44,  1.18s/it]Dataset Size: 1%, Epoch: 811, Train Loss: 9.14057970046997, Val Loss: 11.009334180262181

Epoch Progress:  81%|███████████████████▍    | 811/1000 [15:59<03:43,  1.18s/it]
Epoch Progress:  81%|███████████████████▍    | 812/1000 [16:00<03:41,  1.18s/it]
Epoch Progress:  81%|███████████████████▌    | 813/1000 [16:02<03:41,  1.18s/it]
Epoch Progress:  81%|███████████████████▌    | 814/1000 [16:03<03:40,  1.18s/it]
Epoch Progress:  82%|███████████████████▌    | 815/1000 [16:04<03:38,  1.18s/it]
Epoch Progress:  82%|███████████████████▌    | 816/1000 [16:05<03:37,  1.18s/it]
Epoch Progress:  82%|███████████████████▌    | 817/1000 [16:06<03:35,  1.18s/it]
Epoch Progress:  82%|███████████████████▋    | 818/1000 [16:08<03:34,  1.18s/it]
Epoch Progress:  82%|███████████████████▋    | 819/1000 [16:09<03:33,  1.18s/it]
Epoch Progress:  82%|███████████████████▋    | 820/1000 [16:10<03:32,  1.18s/it]Dataset Size: 1%, Epoch: 821, Train Loss: 9.139437556266785, Val Loss: 11.009918683535092

Epoch Progress:  82%|███████████████████▋    | 821/1000 [16:11<03:33,  1.19s/it]
Epoch Progress:  82%|███████████████████▋    | 822/1000 [16:12<03:31,  1.19s/it]
Epoch Progress:  82%|███████████████████▊    | 823/1000 [16:13<03:29,  1.19s/it]
Epoch Progress:  82%|███████████████████▊    | 824/1000 [16:15<03:28,  1.18s/it]
Epoch Progress:  82%|███████████████████▊    | 825/1000 [16:16<03:26,  1.18s/it]
Epoch Progress:  83%|███████████████████▊    | 826/1000 [16:17<03:25,  1.18s/it]
Epoch Progress:  83%|███████████████████▊    | 827/1000 [16:18<03:24,  1.18s/it]
Epoch Progress:  83%|███████████████████▊    | 828/1000 [16:19<03:23,  1.18s/it]
Epoch Progress:  83%|███████████████████▉    | 829/1000 [16:21<03:21,  1.18s/it]
Epoch Progress:  83%|███████████████████▉    | 830/1000 [16:22<03:21,  1.18s/it]Dataset Size: 1%, Epoch: 831, Train Loss: 9.142634272575378, Val Loss: 11.00855980910264

Epoch Progress:  83%|███████████████████▉    | 831/1000 [16:23<03:19,  1.18s/it]
Epoch Progress:  83%|███████████████████▉    | 832/1000 [16:24<03:18,  1.18s/it]
Epoch Progress:  83%|███████████████████▉    | 833/1000 [16:25<03:17,  1.18s/it]
Epoch Progress:  83%|████████████████████    | 834/1000 [16:26<03:15,  1.18s/it]
Epoch Progress:  84%|████████████████████    | 835/1000 [16:28<03:14,  1.18s/it]
Epoch Progress:  84%|████████████████████    | 836/1000 [16:29<03:13,  1.18s/it]
Epoch Progress:  84%|████████████████████    | 837/1000 [16:30<03:12,  1.18s/it]
Epoch Progress:  84%|████████████████████    | 838/1000 [16:31<03:11,  1.18s/it]
Epoch Progress:  84%|████████████████████▏   | 839/1000 [16:32<03:10,  1.18s/it]
Epoch Progress:  84%|████████████████████▏   | 840/1000 [16:34<03:08,  1.18s/it]Dataset Size: 1%, Epoch: 841, Train Loss: 9.148202657699585, Val Loss: 11.012455159967596

Epoch Progress:  84%|████████████████████▏   | 841/1000 [16:35<03:07,  1.18s/it]
Epoch Progress:  84%|████████████████████▏   | 842/1000 [16:36<03:06,  1.18s/it]
Epoch Progress:  84%|████████████████████▏   | 843/1000 [16:37<03:05,  1.18s/it]
Epoch Progress:  84%|████████████████████▎   | 844/1000 [16:38<03:04,  1.18s/it]
Epoch Progress:  84%|████████████████████▎   | 845/1000 [16:39<03:03,  1.18s/it]
Epoch Progress:  85%|████████████████████▎   | 846/1000 [16:41<03:02,  1.18s/it]
Epoch Progress:  85%|████████████████████▎   | 847/1000 [16:42<03:01,  1.19s/it]
Epoch Progress:  85%|████████████████████▎   | 848/1000 [16:43<03:01,  1.19s/it]
Epoch Progress:  85%|████████████████████▍   | 849/1000 [16:44<02:59,  1.19s/it]
Epoch Progress:  85%|████████████████████▍   | 850/1000 [16:45<02:58,  1.19s/it]Dataset Size: 1%, Epoch: 851, Train Loss: 9.14077615737915, Val Loss: 11.010462488446917

Epoch Progress:  85%|████████████████████▍   | 851/1000 [16:47<02:56,  1.18s/it]
Epoch Progress:  85%|████████████████████▍   | 852/1000 [16:48<02:55,  1.19s/it]
Epoch Progress:  85%|████████████████████▍   | 853/1000 [16:49<02:54,  1.19s/it]
Epoch Progress:  85%|████████████████████▍   | 854/1000 [16:50<02:52,  1.18s/it]
Epoch Progress:  86%|████████████████████▌   | 855/1000 [16:51<02:52,  1.19s/it]
Epoch Progress:  86%|████████████████████▌   | 856/1000 [16:53<02:50,  1.18s/it]
Epoch Progress:  86%|████████████████████▌   | 857/1000 [16:54<02:48,  1.18s/it]
Epoch Progress:  86%|████████████████████▌   | 858/1000 [16:55<02:47,  1.18s/it]
Epoch Progress:  86%|████████████████████▌   | 859/1000 [16:56<02:46,  1.18s/it]
Epoch Progress:  86%|████████████████████▋   | 860/1000 [16:57<02:44,  1.18s/it]Dataset Size: 1%, Epoch: 861, Train Loss: 9.12595021724701, Val Loss: 11.011789817314643

Epoch Progress:  86%|████████████████████▋   | 861/1000 [16:58<02:43,  1.18s/it]
Epoch Progress:  86%|████████████████████▋   | 862/1000 [17:00<02:42,  1.18s/it]
Epoch Progress:  86%|████████████████████▋   | 863/1000 [17:01<02:41,  1.18s/it]
Epoch Progress:  86%|████████████████████▋   | 864/1000 [17:02<02:40,  1.18s/it]
Epoch Progress:  86%|████████████████████▊   | 865/1000 [17:03<02:39,  1.18s/it]
Epoch Progress:  87%|████████████████████▊   | 866/1000 [17:04<02:38,  1.18s/it]
Epoch Progress:  87%|████████████████████▊   | 867/1000 [17:05<02:37,  1.18s/it]
Epoch Progress:  87%|████████████████████▊   | 868/1000 [17:07<02:35,  1.18s/it]
Epoch Progress:  87%|████████████████████▊   | 869/1000 [17:08<02:34,  1.18s/it]
Epoch Progress:  87%|████████████████████▉   | 870/1000 [17:09<02:33,  1.18s/it]Dataset Size: 1%, Epoch: 871, Train Loss: 9.110360026359558, Val Loss: 11.011005302528282

Epoch Progress:  87%|████████████████████▉   | 871/1000 [17:10<02:32,  1.18s/it]
Epoch Progress:  87%|████████████████████▉   | 872/1000 [17:11<02:31,  1.18s/it]
Epoch Progress:  87%|████████████████████▉   | 873/1000 [17:13<02:30,  1.18s/it]
Epoch Progress:  87%|████████████████████▉   | 874/1000 [17:14<02:29,  1.19s/it]
Epoch Progress:  88%|█████████████████████   | 875/1000 [17:15<02:28,  1.19s/it]
Epoch Progress:  88%|█████████████████████   | 876/1000 [17:16<02:26,  1.18s/it]
Epoch Progress:  88%|█████████████████████   | 877/1000 [17:17<02:25,  1.19s/it]
Epoch Progress:  88%|█████████████████████   | 878/1000 [17:18<02:24,  1.18s/it]
Epoch Progress:  88%|█████████████████████   | 879/1000 [17:20<02:23,  1.18s/it]
Epoch Progress:  88%|█████████████████████   | 880/1000 [17:21<02:21,  1.18s/it]Dataset Size: 1%, Epoch: 881, Train Loss: 9.130889058113098, Val Loss: 11.014944361401843

Epoch Progress:  88%|█████████████████████▏  | 881/1000 [17:22<02:21,  1.19s/it]
Epoch Progress:  88%|█████████████████████▏  | 882/1000 [17:23<02:19,  1.19s/it]
Epoch Progress:  88%|█████████████████████▏  | 883/1000 [17:24<02:18,  1.18s/it]
Epoch Progress:  88%|█████████████████████▏  | 884/1000 [17:26<02:17,  1.18s/it]
Epoch Progress:  88%|█████████████████████▏  | 885/1000 [17:27<02:15,  1.18s/it]
Epoch Progress:  89%|█████████████████████▎  | 886/1000 [17:28<02:14,  1.18s/it]
Epoch Progress:  89%|█████████████████████▎  | 887/1000 [17:29<02:13,  1.18s/it]
Epoch Progress:  89%|█████████████████████▎  | 888/1000 [17:30<02:12,  1.18s/it]
Epoch Progress:  89%|█████████████████████▎  | 889/1000 [17:32<02:11,  1.19s/it]
Epoch Progress:  89%|█████████████████████▎  | 890/1000 [17:33<02:10,  1.18s/it]Dataset Size: 1%, Epoch: 891, Train Loss: 9.118276119232178, Val Loss: 11.012568610055107

Epoch Progress:  89%|█████████████████████▍  | 891/1000 [17:34<02:08,  1.18s/it]
Epoch Progress:  89%|█████████████████████▍  | 892/1000 [17:35<02:07,  1.18s/it]
Epoch Progress:  89%|█████████████████████▍  | 893/1000 [17:36<02:06,  1.18s/it]
Epoch Progress:  89%|█████████████████████▍  | 894/1000 [17:37<02:05,  1.18s/it]
Epoch Progress:  90%|█████████████████████▍  | 895/1000 [17:39<02:03,  1.18s/it]
Epoch Progress:  90%|█████████████████████▌  | 896/1000 [17:40<02:02,  1.18s/it]
Epoch Progress:  90%|█████████████████████▌  | 897/1000 [17:41<02:01,  1.18s/it]
Epoch Progress:  90%|█████████████████████▌  | 898/1000 [17:42<02:00,  1.18s/it]
Epoch Progress:  90%|█████████████████████▌  | 899/1000 [17:43<01:59,  1.18s/it]
Epoch Progress:  90%|█████████████████████▌  | 900/1000 [17:44<01:58,  1.18s/it]Dataset Size: 1%, Epoch: 901, Train Loss: 9.139987111091614, Val Loss: 11.012484092216987

Epoch Progress:  90%|█████████████████████▌  | 901/1000 [17:46<01:57,  1.19s/it]
Epoch Progress:  90%|█████████████████████▋  | 902/1000 [17:47<01:56,  1.19s/it]
Epoch Progress:  90%|█████████████████████▋  | 903/1000 [17:48<01:55,  1.19s/it]
Epoch Progress:  90%|█████████████████████▋  | 904/1000 [17:49<01:54,  1.19s/it]
Epoch Progress:  90%|█████████████████████▋  | 905/1000 [17:50<01:52,  1.19s/it]
Epoch Progress:  91%|█████████████████████▋  | 906/1000 [17:52<01:51,  1.19s/it]
Epoch Progress:  91%|█████████████████████▊  | 907/1000 [17:53<01:50,  1.19s/it]
Epoch Progress:  91%|█████████████████████▊  | 908/1000 [17:54<01:49,  1.19s/it]
Epoch Progress:  91%|█████████████████████▊  | 909/1000 [17:55<01:47,  1.18s/it]
Epoch Progress:  91%|█████████████████████▊  | 910/1000 [17:56<01:46,  1.18s/it]Dataset Size: 1%, Epoch: 911, Train Loss: 9.140103459358215, Val Loss: 11.010508834541618

Epoch Progress:  91%|█████████████████████▊  | 911/1000 [17:58<01:45,  1.18s/it]
Epoch Progress:  91%|█████████████████████▉  | 912/1000 [17:59<01:44,  1.18s/it]
Epoch Progress:  91%|█████████████████████▉  | 913/1000 [18:00<01:42,  1.18s/it]
Epoch Progress:  91%|█████████████████████▉  | 914/1000 [18:01<01:41,  1.18s/it]
Epoch Progress:  92%|█████████████████████▉  | 915/1000 [18:02<01:40,  1.18s/it]
Epoch Progress:  92%|█████████████████████▉  | 916/1000 [18:03<01:39,  1.18s/it]
Epoch Progress:  92%|██████████████████████  | 917/1000 [18:05<01:38,  1.18s/it]
Epoch Progress:  92%|██████████████████████  | 918/1000 [18:06<01:36,  1.18s/it]
Epoch Progress:  92%|██████████████████████  | 919/1000 [18:07<01:35,  1.18s/it]
Epoch Progress:  92%|██████████████████████  | 920/1000 [18:08<01:34,  1.18s/it]Dataset Size: 1%, Epoch: 921, Train Loss: 9.096235871315002, Val Loss: 11.011496655352705

Epoch Progress:  92%|██████████████████████  | 921/1000 [18:09<01:33,  1.18s/it]
Epoch Progress:  92%|██████████████████████▏ | 922/1000 [18:11<01:32,  1.18s/it]
Epoch Progress:  92%|██████████████████████▏ | 923/1000 [18:12<01:30,  1.18s/it]
Epoch Progress:  92%|██████████████████████▏ | 924/1000 [18:13<01:29,  1.18s/it]
Epoch Progress:  92%|██████████████████████▏ | 925/1000 [18:14<01:28,  1.18s/it]
Epoch Progress:  93%|██████████████████████▏ | 926/1000 [18:15<01:27,  1.18s/it]
Epoch Progress:  93%|██████████████████████▏ | 927/1000 [18:16<01:26,  1.19s/it]
Epoch Progress:  93%|██████████████████████▎ | 928/1000 [18:18<01:25,  1.19s/it]
Epoch Progress:  93%|██████████████████████▎ | 929/1000 [18:19<01:24,  1.18s/it]
Epoch Progress:  93%|██████████████████████▎ | 930/1000 [18:20<01:22,  1.18s/it]Dataset Size: 1%, Epoch: 931, Train Loss: 9.144653558731079, Val Loss: 11.012745634301917

Epoch Progress:  93%|██████████████████████▎ | 931/1000 [18:21<01:21,  1.19s/it]
Epoch Progress:  93%|██████████████████████▎ | 932/1000 [18:22<01:20,  1.18s/it]
Epoch Progress:  93%|██████████████████████▍ | 933/1000 [18:24<01:19,  1.19s/it]
Epoch Progress:  93%|██████████████████████▍ | 934/1000 [18:25<01:18,  1.19s/it]
Epoch Progress:  94%|██████████████████████▍ | 935/1000 [18:26<01:16,  1.18s/it]
Epoch Progress:  94%|██████████████████████▍ | 936/1000 [18:27<01:15,  1.18s/it]
Epoch Progress:  94%|██████████████████████▍ | 937/1000 [18:28<01:14,  1.18s/it]
Epoch Progress:  94%|██████████████████████▌ | 938/1000 [18:29<01:13,  1.18s/it]
Epoch Progress:  94%|██████████████████████▌ | 939/1000 [18:31<01:11,  1.18s/it]
Epoch Progress:  94%|██████████████████████▌ | 940/1000 [18:32<01:10,  1.18s/it]Dataset Size: 1%, Epoch: 941, Train Loss: 9.130800366401672, Val Loss: 11.0111239544757

Epoch Progress:  94%|██████████████████████▌ | 941/1000 [18:33<01:09,  1.18s/it]
Epoch Progress:  94%|██████████████████████▌ | 942/1000 [18:34<01:08,  1.18s/it]
Epoch Progress:  94%|██████████████████████▋ | 943/1000 [18:35<01:07,  1.18s/it]
Epoch Progress:  94%|██████████████████████▋ | 944/1000 [18:37<01:06,  1.18s/it]
Epoch Progress:  94%|██████████████████████▋ | 945/1000 [18:38<01:04,  1.18s/it]
Epoch Progress:  95%|██████████████████████▋ | 946/1000 [18:39<01:03,  1.18s/it]
Epoch Progress:  95%|██████████████████████▋ | 947/1000 [18:40<01:02,  1.18s/it]
Epoch Progress:  95%|██████████████████████▊ | 948/1000 [18:41<01:01,  1.18s/it]
Epoch Progress:  95%|██████████████████████▊ | 949/1000 [18:42<01:00,  1.18s/it]
Epoch Progress:  95%|██████████████████████▊ | 950/1000 [18:44<00:59,  1.18s/it]Dataset Size: 1%, Epoch: 951, Train Loss: 9.136134266853333, Val Loss: 11.013742843231597

Epoch Progress:  95%|██████████████████████▊ | 951/1000 [18:45<00:57,  1.18s/it]
Epoch Progress:  95%|██████████████████████▊ | 952/1000 [18:46<00:56,  1.18s/it]
Epoch Progress:  95%|██████████████████████▊ | 953/1000 [18:47<00:55,  1.18s/it]
Epoch Progress:  95%|██████████████████████▉ | 954/1000 [18:48<00:54,  1.19s/it]
Epoch Progress:  96%|██████████████████████▉ | 955/1000 [18:50<00:53,  1.18s/it]
Epoch Progress:  96%|██████████████████████▉ | 956/1000 [18:51<00:52,  1.18s/it]
Epoch Progress:  96%|██████████████████████▉ | 957/1000 [18:52<00:51,  1.19s/it]
Epoch Progress:  96%|██████████████████████▉ | 958/1000 [18:53<00:49,  1.19s/it]
Epoch Progress:  96%|███████████████████████ | 959/1000 [18:54<00:48,  1.18s/it]
Epoch Progress:  96%|███████████████████████ | 960/1000 [18:56<00:47,  1.18s/it]Dataset Size: 1%, Epoch: 961, Train Loss: 9.109071254730225, Val Loss: 11.012853089865152

Epoch Progress:  96%|███████████████████████ | 961/1000 [18:57<00:46,  1.18s/it]
Epoch Progress:  96%|███████████████████████ | 962/1000 [18:58<00:44,  1.18s/it]
Epoch Progress:  96%|███████████████████████ | 963/1000 [18:59<00:43,  1.18s/it]
Epoch Progress:  96%|███████████████████████▏| 964/1000 [19:00<00:42,  1.18s/it]
Epoch Progress:  96%|███████████████████████▏| 965/1000 [19:01<00:41,  1.18s/it]
Epoch Progress:  97%|███████████████████████▏| 966/1000 [19:03<00:40,  1.18s/it]
Epoch Progress:  97%|███████████████████████▏| 967/1000 [19:04<00:39,  1.18s/it]
Epoch Progress:  97%|███████████████████████▏| 968/1000 [19:05<00:37,  1.18s/it]
Epoch Progress:  97%|███████████████████████▎| 969/1000 [19:06<00:36,  1.18s/it]
Epoch Progress:  97%|███████████████████████▎| 970/1000 [19:07<00:35,  1.18s/it]Dataset Size: 1%, Epoch: 971, Train Loss: 9.119047403335571, Val Loss: 11.011850852470893

Epoch Progress:  97%|███████████████████████▎| 971/1000 [19:08<00:34,  1.18s/it]
Epoch Progress:  97%|███████████████████████▎| 972/1000 [19:10<00:32,  1.18s/it]
Epoch Progress:  97%|███████████████████████▎| 973/1000 [19:11<00:31,  1.18s/it]
Epoch Progress:  97%|███████████████████████▍| 974/1000 [19:12<00:30,  1.18s/it]
Epoch Progress:  98%|███████████████████████▍| 975/1000 [19:13<00:29,  1.18s/it]
Epoch Progress:  98%|███████████████████████▍| 976/1000 [19:14<00:28,  1.18s/it]
Epoch Progress:  98%|███████████████████████▍| 977/1000 [19:16<00:27,  1.18s/it]
Epoch Progress:  98%|███████████████████████▍| 978/1000 [19:17<00:25,  1.18s/it]
Epoch Progress:  98%|███████████████████████▍| 979/1000 [19:18<00:24,  1.18s/it]
Epoch Progress:  98%|███████████████████████▌| 980/1000 [19:19<00:23,  1.19s/it]Dataset Size: 1%, Epoch: 981, Train Loss: 9.1276935338974, Val Loss: 11.011541614284763

Epoch Progress:  98%|███████████████████████▌| 981/1000 [19:20<00:22,  1.19s/it]
Epoch Progress:  98%|███████████████████████▌| 982/1000 [19:21<00:21,  1.19s/it]
Epoch Progress:  98%|███████████████████████▌| 983/1000 [19:23<00:20,  1.18s/it]
Epoch Progress:  98%|███████████████████████▌| 984/1000 [19:24<00:18,  1.18s/it]
Epoch Progress:  98%|███████████████████████▋| 985/1000 [19:25<00:17,  1.18s/it]
Epoch Progress:  99%|███████████████████████▋| 986/1000 [19:26<00:16,  1.18s/it]
Epoch Progress:  99%|███████████████████████▋| 987/1000 [19:27<00:15,  1.18s/it]
Epoch Progress:  99%|███████████████████████▋| 988/1000 [19:29<00:14,  1.18s/it]
Epoch Progress:  99%|███████████████████████▋| 989/1000 [19:30<00:12,  1.18s/it]
Epoch Progress:  99%|███████████████████████▊| 990/1000 [19:31<00:11,  1.18s/it]Dataset Size: 1%, Epoch: 991, Train Loss: 9.108376860618591, Val Loss: 11.011671809407023

Epoch Progress:  99%|███████████████████████▊| 991/1000 [19:32<00:10,  1.18s/it]
Epoch Progress:  99%|███████████████████████▊| 992/1000 [19:33<00:09,  1.18s/it]
Epoch Progress:  99%|███████████████████████▊| 993/1000 [19:34<00:08,  1.18s/it]
Epoch Progress:  99%|███████████████████████▊| 994/1000 [19:36<00:07,  1.18s/it]
Epoch Progress: 100%|███████████████████████▉| 995/1000 [19:37<00:05,  1.18s/it]
Epoch Progress: 100%|███████████████████████▉| 996/1000 [19:38<00:04,  1.18s/it]
Epoch Progress: 100%|███████████████████████▉| 997/1000 [19:39<00:03,  1.18s/it]
Epoch Progress: 100%|███████████████████████▉| 998/1000 [19:40<00:02,  1.18s/it]
Epoch Progress: 100%|███████████████████████▉| 999/1000 [19:42<00:01,  1.18s/it]
Epoch Progress: 100%|███████████████████████| 1000/1000 [19:43<00:00,  1.18s/it]
Dataset Size: 1%, Val Perplexity: 44179.12890625
'''

# Use regex to extract train and validation losses every 10 epochs.
pattern = r"Epoch: (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)"
matches = re.findall(pattern, log_data)

# Convert matches to a DataFrame.
loss_data = [(int(epoch), float(train_loss), float(val_loss)) for epoch, train_loss, val_loss in matches]
df_losses = pd.DataFrame(loss_data, columns=['Epoch', 'Train Loss', 'Val Loss'])

# Filter for every 10 epochs.
df_losses_filtered = df_losses[df_losses['Epoch'] % 10 == 1].reset_index(drop=True)
import matplotlib.pyplot as plt

# Extracting epochs, training losses, and validation losses
epochs = df_losses_filtered['Epoch']
train_losses = df_losses_filtered['Train Loss']
val_losses = df_losses_filtered['Val Loss']

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=3)  
plt.plot(epochs, val_losses, label='Validation Loss', marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Params=1.6M, Fraction=0.01, LR=0.001')
plt.legend()
plt.grid(True)
plt.savefig('1.6m_small_f_0.01_lr_0.001.png')
plt.show()