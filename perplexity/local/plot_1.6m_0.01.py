import re
import pandas as pd

log_data = '''
Data Iteration:   0%|                                     | 0/1 [00:00<?, ?it/s]
Model is on device cuda and has 1622688 parameters

Epoch Progress:   0%|                                  | 0/1000 [00:00<?, ?it/s]Dataset Size: 1%, Epoch: 1, Train Loss: 11.004348516464233, Val Loss: 10.997926981021196

Epoch Progress:   0%|                          | 1/1000 [00:01<24:53,  1.49s/it]
Epoch Progress:   0%|                          | 2/1000 [00:02<21:01,  1.26s/it]
Epoch Progress:   0%|                          | 3/1000 [00:03<19:53,  1.20s/it]
Epoch Progress:   0%|                          | 4/1000 [00:04<19:19,  1.16s/it]
Epoch Progress:   0%|▏                         | 5/1000 [00:06<19:27,  1.17s/it]
Epoch Progress:   1%|▏                         | 6/1000 [00:07<19:03,  1.15s/it]
Epoch Progress:   1%|▏                         | 7/1000 [00:08<18:44,  1.13s/it]
Epoch Progress:   1%|▏                         | 8/1000 [00:09<18:35,  1.12s/it]
Epoch Progress:   1%|▏                         | 9/1000 [00:10<18:34,  1.12s/it]
Epoch Progress:   1%|▎                        | 10/1000 [00:11<18:29,  1.12s/it]Dataset Size: 1%, Epoch: 11, Train Loss: 10.976128101348877, Val Loss: 10.97914624825502

Epoch Progress:   1%|▎                        | 11/1000 [00:12<18:28,  1.12s/it]
Epoch Progress:   1%|▎                        | 12/1000 [00:13<18:24,  1.12s/it]
Epoch Progress:   1%|▎                        | 13/1000 [00:14<18:20,  1.12s/it]
Epoch Progress:   1%|▎                        | 14/1000 [00:16<18:19,  1.12s/it]
Epoch Progress:   2%|▍                        | 15/1000 [00:17<18:36,  1.13s/it]
Epoch Progress:   2%|▍                        | 16/1000 [00:18<18:29,  1.13s/it]
Epoch Progress:   2%|▍                        | 17/1000 [00:19<18:26,  1.13s/it]
Epoch Progress:   2%|▍                        | 18/1000 [00:20<18:23,  1.12s/it]
Epoch Progress:   2%|▍                        | 19/1000 [00:21<18:16,  1.12s/it]
Epoch Progress:   2%|▌                        | 20/1000 [00:22<18:29,  1.13s/it]Dataset Size: 1%, Epoch: 21, Train Loss: 10.886632442474365, Val Loss: 10.924477895100912

Epoch Progress:   2%|▌                        | 21/1000 [00:23<18:19,  1.12s/it]
Epoch Progress:   2%|▌                        | 22/1000 [00:25<18:19,  1.12s/it]
Epoch Progress:   2%|▌                        | 23/1000 [00:26<18:17,  1.12s/it]
Epoch Progress:   2%|▌                        | 24/1000 [00:27<18:13,  1.12s/it]
Epoch Progress:   2%|▋                        | 25/1000 [00:28<18:07,  1.12s/it]
Epoch Progress:   3%|▋                        | 26/1000 [00:29<18:00,  1.11s/it]
Epoch Progress:   3%|▋                        | 27/1000 [00:30<18:05,  1.12s/it]
Epoch Progress:   3%|▋                        | 28/1000 [00:31<18:01,  1.11s/it]
Epoch Progress:   3%|▋                        | 29/1000 [00:32<17:58,  1.11s/it]
Epoch Progress:   3%|▊                        | 30/1000 [00:33<17:57,  1.11s/it]Dataset Size: 1%, Epoch: 31, Train Loss: 10.671182870864868, Val Loss: 10.780084341000288

Epoch Progress:   3%|▊                        | 31/1000 [00:35<17:55,  1.11s/it]
Epoch Progress:   3%|▊                        | 32/1000 [00:36<17:53,  1.11s/it]
Epoch Progress:   3%|▊                        | 33/1000 [00:37<17:58,  1.12s/it]
Epoch Progress:   3%|▊                        | 34/1000 [00:38<17:56,  1.11s/it]
Epoch Progress:   4%|▉                        | 35/1000 [00:39<18:32,  1.15s/it]
Epoch Progress:   4%|▉                        | 36/1000 [00:40<18:20,  1.14s/it]
Epoch Progress:   4%|▉                        | 37/1000 [00:41<18:06,  1.13s/it]
Epoch Progress:   4%|▉                        | 38/1000 [00:42<17:58,  1.12s/it]
Epoch Progress:   4%|▉                        | 39/1000 [00:44<17:49,  1.11s/it]
Epoch Progress:   4%|█                        | 40/1000 [00:45<17:42,  1.11s/it]Dataset Size: 1%, Epoch: 41, Train Loss: 10.148192405700684, Val Loss: 10.403442749610313

Epoch Progress:   4%|█                        | 41/1000 [00:46<17:41,  1.11s/it]
Epoch Progress:   4%|█                        | 42/1000 [00:47<17:43,  1.11s/it]
Epoch Progress:   4%|█                        | 43/1000 [00:48<17:42,  1.11s/it]
Epoch Progress:   4%|█                        | 44/1000 [00:49<17:43,  1.11s/it]
Epoch Progress:   4%|█▏                       | 45/1000 [00:50<17:59,  1.13s/it]
Epoch Progress:   5%|█▏                       | 46/1000 [00:51<17:48,  1.12s/it]
Epoch Progress:   5%|█▏                       | 47/1000 [00:52<17:43,  1.12s/it]
Epoch Progress:   5%|█▏                       | 48/1000 [00:54<17:36,  1.11s/it]
Epoch Progress:   5%|█▏                       | 49/1000 [00:55<17:35,  1.11s/it]
Epoch Progress:   5%|█▎                       | 50/1000 [00:56<17:51,  1.13s/it]Dataset Size: 1%, Epoch: 51, Train Loss: 9.363978862762451, Val Loss: 9.843058952918419

Epoch Progress:   5%|█▎                       | 51/1000 [00:57<17:46,  1.12s/it]
Epoch Progress:   5%|█▎                       | 52/1000 [00:58<17:42,  1.12s/it]
Epoch Progress:   5%|█▎                       | 53/1000 [00:59<17:40,  1.12s/it]
Epoch Progress:   5%|█▎                       | 54/1000 [01:00<17:37,  1.12s/it]
Epoch Progress:   6%|█▍                       | 55/1000 [01:01<17:49,  1.13s/it]
Epoch Progress:   6%|█▍                       | 56/1000 [01:03<17:41,  1.12s/it]
Epoch Progress:   6%|█▍                       | 57/1000 [01:04<17:35,  1.12s/it]
Epoch Progress:   6%|█▍                       | 58/1000 [01:05<17:30,  1.12s/it]
Epoch Progress:   6%|█▍                       | 59/1000 [01:06<17:24,  1.11s/it]
Epoch Progress:   6%|█▌                       | 60/1000 [01:07<17:22,  1.11s/it]Dataset Size: 1%, Epoch: 61, Train Loss: 8.580154418945312, Val Loss: 9.314748910757212

Epoch Progress:   6%|█▌                       | 61/1000 [01:08<17:23,  1.11s/it]
Epoch Progress:   6%|█▌                       | 62/1000 [01:09<17:21,  1.11s/it]
Epoch Progress:   6%|█▌                       | 63/1000 [01:10<17:18,  1.11s/it]
Epoch Progress:   6%|█▌                       | 64/1000 [01:11<17:15,  1.11s/it]
Epoch Progress:   6%|█▋                       | 65/1000 [01:13<17:12,  1.10s/it]
Epoch Progress:   7%|█▋                       | 66/1000 [01:14<17:09,  1.10s/it]
Epoch Progress:   7%|█▋                       | 67/1000 [01:15<17:14,  1.11s/it]
Epoch Progress:   7%|█▋                       | 68/1000 [01:16<17:11,  1.11s/it]
Epoch Progress:   7%|█▋                       | 69/1000 [01:17<17:09,  1.11s/it]
Epoch Progress:   7%|█▊                       | 70/1000 [01:18<17:09,  1.11s/it]Dataset Size: 1%, Epoch: 71, Train Loss: 7.891816020011902, Val Loss: 8.877614388099083

Epoch Progress:   7%|█▊                       | 71/1000 [01:19<17:07,  1.11s/it]
Epoch Progress:   7%|█▊                       | 72/1000 [01:20<17:08,  1.11s/it]
Epoch Progress:   7%|█▊                       | 73/1000 [01:21<17:08,  1.11s/it]
Epoch Progress:   7%|█▊                       | 74/1000 [01:23<17:09,  1.11s/it]
Epoch Progress:   8%|█▉                       | 75/1000 [01:24<17:25,  1.13s/it]
Epoch Progress:   8%|█▉                       | 76/1000 [01:25<17:18,  1.12s/it]
Epoch Progress:   8%|█▉                       | 77/1000 [01:26<17:11,  1.12s/it]
Epoch Progress:   8%|█▉                       | 78/1000 [01:27<17:05,  1.11s/it]
Epoch Progress:   8%|█▉                       | 79/1000 [01:28<17:03,  1.11s/it]
Epoch Progress:   8%|██                       | 80/1000 [01:29<17:18,  1.13s/it]Dataset Size: 1%, Epoch: 81, Train Loss: 7.291096568107605, Val Loss: 8.532154352237017

Epoch Progress:   8%|██                       | 81/1000 [01:30<17:13,  1.12s/it]
Epoch Progress:   8%|██                       | 82/1000 [01:31<17:08,  1.12s/it]
Epoch Progress:   8%|██                       | 83/1000 [01:33<17:02,  1.11s/it]
Epoch Progress:   8%|██                       | 84/1000 [01:34<17:01,  1.11s/it]
Epoch Progress:   8%|██▏                      | 85/1000 [01:35<16:54,  1.11s/it]
Epoch Progress:   9%|██▏                      | 86/1000 [01:36<16:51,  1.11s/it]
Epoch Progress:   9%|██▏                      | 87/1000 [01:37<16:49,  1.11s/it]
Epoch Progress:   9%|██▏                      | 88/1000 [01:38<16:45,  1.10s/it]
Epoch Progress:   9%|██▏                      | 89/1000 [01:39<16:43,  1.10s/it]
Epoch Progress:   9%|██▎                      | 90/1000 [01:40<17:07,  1.13s/it]Dataset Size: 1%, Epoch: 91, Train Loss: 6.8147324323654175, Val Loss: 8.301772753397623

Epoch Progress:   9%|██▎                      | 91/1000 [01:41<16:57,  1.12s/it]
Epoch Progress:   9%|██▎                      | 92/1000 [01:43<16:52,  1.11s/it]
Epoch Progress:   9%|██▎                      | 93/1000 [01:44<16:47,  1.11s/it]
Epoch Progress:   9%|██▎                      | 94/1000 [01:45<16:42,  1.11s/it]
Epoch Progress:  10%|██▍                      | 95/1000 [01:46<17:25,  1.16s/it]
Epoch Progress:  10%|██▍                      | 96/1000 [01:47<17:11,  1.14s/it]
Epoch Progress:  10%|██▍                      | 97/1000 [01:48<17:03,  1.13s/it]
Epoch Progress:  10%|██▍                      | 98/1000 [01:49<16:54,  1.12s/it]
Epoch Progress:  10%|██▍                      | 99/1000 [01:50<16:45,  1.12s/it]
Epoch Progress:  10%|██▍                     | 100/1000 [01:52<16:39,  1.11s/it]Dataset Size: 1%, Epoch: 101, Train Loss: 6.4889678955078125, Val Loss: 8.213376069680239

Epoch Progress:  10%|██▍                     | 101/1000 [01:53<16:42,  1.11s/it]
Epoch Progress:  10%|██▍                     | 102/1000 [01:54<16:39,  1.11s/it]
Epoch Progress:  10%|██▍                     | 103/1000 [01:55<16:35,  1.11s/it]
Epoch Progress:  10%|██▍                     | 104/1000 [01:56<16:32,  1.11s/it]
Epoch Progress:  10%|██▌                     | 105/1000 [01:57<16:50,  1.13s/it]
Epoch Progress:  11%|██▌                     | 106/1000 [01:58<16:34,  1.11s/it]
Epoch Progress:  11%|██▌                     | 107/1000 [01:59<16:38,  1.12s/it]
Epoch Progress:  11%|██▌                     | 108/1000 [02:01<16:37,  1.12s/it]
Epoch Progress:  11%|██▌                     | 109/1000 [02:02<16:36,  1.12s/it]
Epoch Progress:  11%|██▋                     | 110/1000 [02:03<17:03,  1.15s/it]Dataset Size: 1%, Epoch: 111, Train Loss: 6.262245059013367, Val Loss: 8.191175534174992

Epoch Progress:  11%|██▋                     | 111/1000 [02:04<16:50,  1.14s/it]
Epoch Progress:  11%|██▋                     | 112/1000 [02:05<16:43,  1.13s/it]
Epoch Progress:  11%|██▋                     | 113/1000 [02:06<16:26,  1.11s/it]
Epoch Progress:  11%|██▋                     | 114/1000 [02:07<16:26,  1.11s/it]
Epoch Progress:  12%|██▊                     | 115/1000 [02:08<16:22,  1.11s/it]
Epoch Progress:  12%|██▊                     | 116/1000 [02:09<16:09,  1.10s/it]
Epoch Progress:  12%|██▊                     | 117/1000 [02:11<15:59,  1.09s/it]
Epoch Progress:  12%|██▊                     | 118/1000 [02:12<16:08,  1.10s/it]
Epoch Progress:  12%|██▊                     | 119/1000 [02:13<15:58,  1.09s/it]
Epoch Progress:  12%|██▉                     | 120/1000 [02:14<15:59,  1.09s/it]Dataset Size: 1%, Epoch: 121, Train Loss: 6.03608250617981, Val Loss: 8.174722585922632

Epoch Progress:  12%|██▉                     | 121/1000 [02:15<16:00,  1.09s/it]
Epoch Progress:  12%|██▉                     | 122/1000 [02:16<15:59,  1.09s/it]
Epoch Progress:  12%|██▉                     | 123/1000 [02:17<16:01,  1.10s/it]
Epoch Progress:  12%|██▉                     | 124/1000 [02:18<15:51,  1.09s/it]
Epoch Progress:  12%|███                     | 125/1000 [02:19<16:38,  1.14s/it]
Epoch Progress:  13%|███                     | 126/1000 [02:20<16:18,  1.12s/it]
Epoch Progress:  13%|███                     | 127/1000 [02:22<16:17,  1.12s/it]
Epoch Progress:  13%|███                     | 128/1000 [02:23<16:12,  1.12s/it]
Epoch Progress:  13%|███                     | 129/1000 [02:24<15:57,  1.10s/it]
Epoch Progress:  13%|███                     | 130/1000 [02:25<15:46,  1.09s/it]Dataset Size: 1%, Epoch: 131, Train Loss: 5.758623838424683, Val Loss: 8.164579880543243

Epoch Progress:  13%|███▏                    | 131/1000 [02:26<15:48,  1.09s/it]
Epoch Progress:  13%|███▏                    | 132/1000 [02:27<15:50,  1.10s/it]
Epoch Progress:  13%|███▏                    | 133/1000 [02:28<15:51,  1.10s/it]
Epoch Progress:  13%|███▏                    | 134/1000 [02:29<15:40,  1.09s/it]
Epoch Progress:  14%|███▏                    | 135/1000 [02:30<15:35,  1.08s/it]
Epoch Progress:  14%|███▎                    | 136/1000 [02:31<15:28,  1.07s/it]
Epoch Progress:  14%|███▎                    | 137/1000 [02:32<15:23,  1.07s/it]
Epoch Progress:  14%|███▎                    | 138/1000 [02:33<15:20,  1.07s/it]
Epoch Progress:  14%|███▎                    | 139/1000 [02:35<15:16,  1.06s/it]
Epoch Progress:  14%|███▎                    | 140/1000 [02:36<15:13,  1.06s/it]Dataset Size: 1%, Epoch: 141, Train Loss: 5.44551956653595, Val Loss: 8.170904379624586

Epoch Progress:  14%|███▍                    | 141/1000 [02:37<15:13,  1.06s/it]
Epoch Progress:  14%|███▍                    | 142/1000 [02:38<15:11,  1.06s/it]
Epoch Progress:  14%|███▍                    | 143/1000 [02:39<15:09,  1.06s/it]
Epoch Progress:  14%|███▍                    | 144/1000 [02:40<15:11,  1.07s/it]
Epoch Progress:  14%|███▍                    | 145/1000 [02:41<15:09,  1.06s/it]
Epoch Progress:  15%|███▌                    | 146/1000 [02:42<15:07,  1.06s/it]
Epoch Progress:  15%|███▌                    | 147/1000 [02:43<15:06,  1.06s/it]
Epoch Progress:  15%|███▌                    | 148/1000 [02:44<15:09,  1.07s/it]
Epoch Progress:  15%|███▌                    | 149/1000 [02:45<15:06,  1.07s/it]
Epoch Progress:  15%|███▌                    | 150/1000 [02:46<15:03,  1.06s/it]Dataset Size: 1%, Epoch: 151, Train Loss: 5.136634349822998, Val Loss: 8.205261621719751

Epoch Progress:  15%|███▌                    | 151/1000 [02:47<15:01,  1.06s/it]
Epoch Progress:  15%|███▋                    | 152/1000 [02:48<15:00,  1.06s/it]
Epoch Progress:  15%|███▋                    | 153/1000 [02:49<15:00,  1.06s/it]
Epoch Progress:  15%|███▋                    | 154/1000 [02:50<15:00,  1.06s/it]
Epoch Progress:  16%|███▋                    | 155/1000 [02:52<14:57,  1.06s/it]
Epoch Progress:  16%|███▋                    | 156/1000 [02:53<14:54,  1.06s/it]
Epoch Progress:  16%|███▊                    | 157/1000 [02:54<14:53,  1.06s/it]
Epoch Progress:  16%|███▊                    | 158/1000 [02:55<14:52,  1.06s/it]
Epoch Progress:  16%|███▊                    | 159/1000 [02:56<14:52,  1.06s/it]
Epoch Progress:  16%|███▊                    | 160/1000 [02:57<14:52,  1.06s/it]Dataset Size: 1%, Epoch: 161, Train Loss: 4.826749920845032, Val Loss: 8.247491824321258

Epoch Progress:  16%|███▊                    | 161/1000 [02:58<14:50,  1.06s/it]
Epoch Progress:  16%|███▉                    | 162/1000 [02:59<14:49,  1.06s/it]
Epoch Progress:  16%|███▉                    | 163/1000 [03:00<14:52,  1.07s/it]
Epoch Progress:  16%|███▉                    | 164/1000 [03:01<14:49,  1.06s/it]
Epoch Progress:  16%|███▉                    | 165/1000 [03:02<14:46,  1.06s/it]
Epoch Progress:  17%|███▉                    | 166/1000 [03:03<14:44,  1.06s/it]
Epoch Progress:  17%|████                    | 167/1000 [03:04<14:43,  1.06s/it]
Epoch Progress:  17%|████                    | 168/1000 [03:05<14:41,  1.06s/it]
Epoch Progress:  17%|████                    | 169/1000 [03:06<14:41,  1.06s/it]
Epoch Progress:  17%|████                    | 170/1000 [03:07<14:40,  1.06s/it]Dataset Size: 1%, Epoch: 171, Train Loss: 4.535880923271179, Val Loss: 8.325125963259966

Epoch Progress:  17%|████                    | 171/1000 [03:09<14:40,  1.06s/it]
Epoch Progress:  17%|████▏                   | 172/1000 [03:10<14:42,  1.07s/it]
Epoch Progress:  17%|████▏                   | 173/1000 [03:11<14:40,  1.07s/it]
Epoch Progress:  17%|████▏                   | 174/1000 [03:12<14:38,  1.06s/it]
Epoch Progress:  18%|████▏                   | 175/1000 [03:13<14:37,  1.06s/it]
Epoch Progress:  18%|████▏                   | 176/1000 [03:14<14:34,  1.06s/it]
Epoch Progress:  18%|████▏                   | 177/1000 [03:15<14:36,  1.07s/it]
Epoch Progress:  18%|████▎                   | 178/1000 [03:16<14:34,  1.06s/it]
Epoch Progress:  18%|████▎                   | 179/1000 [03:17<14:32,  1.06s/it]
Epoch Progress:  18%|████▎                   | 180/1000 [03:18<14:31,  1.06s/it]Dataset Size: 1%, Epoch: 181, Train Loss: 4.2691826820373535, Val Loss: 8.428033486390726

Epoch Progress:  18%|████▎                   | 181/1000 [03:19<14:29,  1.06s/it]
Epoch Progress:  18%|████▎                   | 182/1000 [03:20<14:30,  1.06s/it]
Epoch Progress:  18%|████▍                   | 183/1000 [03:21<14:28,  1.06s/it]
Epoch Progress:  18%|████▍                   | 184/1000 [03:22<14:26,  1.06s/it]
Epoch Progress:  18%|████▍                   | 185/1000 [03:23<14:26,  1.06s/it]
Epoch Progress:  19%|████▍                   | 186/1000 [03:24<14:24,  1.06s/it]
Epoch Progress:  19%|████▍                   | 187/1000 [03:26<14:22,  1.06s/it]
Epoch Progress:  19%|████▌                   | 188/1000 [03:27<14:21,  1.06s/it]
Epoch Progress:  19%|████▌                   | 189/1000 [03:28<14:21,  1.06s/it]
Epoch Progress:  19%|████▌                   | 190/1000 [03:29<14:20,  1.06s/it]Dataset Size: 1%, Epoch: 191, Train Loss: 4.018850862979889, Val Loss: 8.5387055568206

Epoch Progress:  19%|████▌                   | 191/1000 [03:30<14:20,  1.06s/it]
Epoch Progress:  19%|████▌                   | 192/1000 [03:31<14:18,  1.06s/it]
Epoch Progress:  19%|████▋                   | 193/1000 [03:32<14:17,  1.06s/it]
Epoch Progress:  19%|████▋                   | 194/1000 [03:33<14:15,  1.06s/it]
Epoch Progress:  20%|████▋                   | 195/1000 [03:34<14:13,  1.06s/it]
Epoch Progress:  20%|████▋                   | 196/1000 [03:35<14:12,  1.06s/it]
Epoch Progress:  20%|████▋                   | 197/1000 [03:36<14:11,  1.06s/it]
Epoch Progress:  20%|████▊                   | 198/1000 [03:37<14:09,  1.06s/it]
Epoch Progress:  20%|████▊                   | 199/1000 [03:38<14:08,  1.06s/it]
Epoch Progress:  20%|████▊                   | 200/1000 [03:39<14:07,  1.06s/it]Dataset Size: 1%, Epoch: 201, Train Loss: 3.792325496673584, Val Loss: 8.651894667209723

Epoch Progress:  20%|████▊                   | 201/1000 [03:40<14:07,  1.06s/it]
Epoch Progress:  20%|████▊                   | 202/1000 [03:41<14:06,  1.06s/it]
Epoch Progress:  20%|████▊                   | 203/1000 [03:42<14:04,  1.06s/it]
Epoch Progress:  20%|████▉                   | 204/1000 [03:44<14:05,  1.06s/it]
Epoch Progress:  20%|████▉                   | 205/1000 [03:45<14:04,  1.06s/it]
Epoch Progress:  21%|████▉                   | 206/1000 [03:46<14:02,  1.06s/it]
Epoch Progress:  21%|████▉                   | 207/1000 [03:47<14:07,  1.07s/it]
Epoch Progress:  21%|████▉                   | 208/1000 [03:48<14:05,  1.07s/it]
Epoch Progress:  21%|█████                   | 209/1000 [03:49<14:03,  1.07s/it]
Epoch Progress:  21%|█████                   | 210/1000 [03:50<14:03,  1.07s/it]Dataset Size: 1%, Epoch: 211, Train Loss: 3.5889047980308533, Val Loss: 8.771975272741074

Epoch Progress:  21%|█████                   | 211/1000 [03:51<14:00,  1.06s/it]
Epoch Progress:  21%|█████                   | 212/1000 [03:52<13:57,  1.06s/it]
Epoch Progress:  21%|█████                   | 213/1000 [03:53<13:55,  1.06s/it]
Epoch Progress:  21%|█████▏                  | 214/1000 [03:54<13:53,  1.06s/it]
Epoch Progress:  22%|█████▏                  | 215/1000 [03:55<13:53,  1.06s/it]
Epoch Progress:  22%|█████▏                  | 216/1000 [03:56<13:51,  1.06s/it]
Epoch Progress:  22%|█████▏                  | 217/1000 [03:57<13:50,  1.06s/it]
Epoch Progress:  22%|█████▏                  | 218/1000 [03:58<13:49,  1.06s/it]
Epoch Progress:  22%|█████▎                  | 219/1000 [04:00<13:52,  1.07s/it]
Epoch Progress:  22%|█████▎                  | 220/1000 [04:01<13:51,  1.07s/it]Dataset Size: 1%, Epoch: 221, Train Loss: 3.4152870178222656, Val Loss: 8.901710436894344

Epoch Progress:  22%|█████▎                  | 221/1000 [04:02<13:48,  1.06s/it]
Epoch Progress:  22%|█████▎                  | 222/1000 [04:03<13:46,  1.06s/it]
Epoch Progress:  22%|█████▎                  | 223/1000 [04:04<13:45,  1.06s/it]
Epoch Progress:  22%|█████▍                  | 224/1000 [04:05<13:45,  1.06s/it]
Epoch Progress:  22%|█████▍                  | 225/1000 [04:06<13:43,  1.06s/it]
Epoch Progress:  23%|█████▍                  | 226/1000 [04:07<13:43,  1.06s/it]
Epoch Progress:  23%|█████▍                  | 227/1000 [04:08<13:41,  1.06s/it]
Epoch Progress:  23%|█████▍                  | 228/1000 [04:09<13:39,  1.06s/it]
Epoch Progress:  23%|█████▍                  | 229/1000 [04:10<13:40,  1.06s/it]
Epoch Progress:  23%|█████▌                  | 230/1000 [04:11<13:38,  1.06s/it]Dataset Size: 1%, Epoch: 231, Train Loss: 3.251246213912964, Val Loss: 9.014221680470001

Epoch Progress:  23%|█████▌                  | 231/1000 [04:12<13:36,  1.06s/it]
Epoch Progress:  23%|█████▌                  | 232/1000 [04:13<13:34,  1.06s/it]
Epoch Progress:  23%|█████▌                  | 233/1000 [04:14<13:32,  1.06s/it]
Epoch Progress:  23%|█████▌                  | 234/1000 [04:15<13:31,  1.06s/it]
Epoch Progress:  24%|█████▋                  | 235/1000 [04:16<13:31,  1.06s/it]
Epoch Progress:  24%|█████▋                  | 236/1000 [04:18<13:29,  1.06s/it]
Epoch Progress:  24%|█████▋                  | 237/1000 [04:19<13:35,  1.07s/it]
Epoch Progress:  24%|█████▋                  | 238/1000 [04:20<13:35,  1.07s/it]
Epoch Progress:  24%|█████▋                  | 239/1000 [04:21<13:32,  1.07s/it]
Epoch Progress:  24%|█████▊                  | 240/1000 [04:22<13:29,  1.07s/it]Dataset Size: 1%, Epoch: 241, Train Loss: 3.1144164204597473, Val Loss: 9.13673210144043

Epoch Progress:  24%|█████▊                  | 241/1000 [04:23<13:27,  1.06s/it]
Epoch Progress:  24%|█████▊                  | 242/1000 [04:24<13:25,  1.06s/it]
Epoch Progress:  24%|█████▊                  | 243/1000 [04:25<13:23,  1.06s/it]
Epoch Progress:  24%|█████▊                  | 244/1000 [04:26<13:22,  1.06s/it]
Epoch Progress:  24%|█████▉                  | 245/1000 [04:27<13:22,  1.06s/it]
Epoch Progress:  25%|█████▉                  | 246/1000 [04:28<13:20,  1.06s/it]
Epoch Progress:  25%|█████▉                  | 247/1000 [04:29<13:19,  1.06s/it]
Epoch Progress:  25%|█████▉                  | 248/1000 [04:30<13:20,  1.06s/it]
Epoch Progress:  25%|█████▉                  | 249/1000 [04:31<13:18,  1.06s/it]
Epoch Progress:  25%|██████                  | 250/1000 [04:32<13:16,  1.06s/it]Dataset Size: 1%, Epoch: 251, Train Loss: 2.989103853702545, Val Loss: 9.267055731553297

Epoch Progress:  25%|██████                  | 251/1000 [04:34<13:14,  1.06s/it]
Epoch Progress:  25%|██████                  | 252/1000 [04:35<13:12,  1.06s/it]
Epoch Progress:  25%|██████                  | 253/1000 [04:36<13:11,  1.06s/it]
Epoch Progress:  25%|██████                  | 254/1000 [04:37<13:10,  1.06s/it]
Epoch Progress:  26%|██████                  | 255/1000 [04:38<13:09,  1.06s/it]
Epoch Progress:  26%|██████▏                 | 256/1000 [04:39<13:08,  1.06s/it]
Epoch Progress:  26%|██████▏                 | 257/1000 [04:40<13:10,  1.06s/it]
Epoch Progress:  26%|██████▏                 | 258/1000 [04:41<13:09,  1.06s/it]
Epoch Progress:  26%|██████▏                 | 259/1000 [04:42<13:08,  1.06s/it]
Epoch Progress:  26%|██████▏                 | 260/1000 [04:43<13:08,  1.07s/it]Dataset Size: 1%, Epoch: 261, Train Loss: 2.8757638335227966, Val Loss: 9.38012267381717

Epoch Progress:  26%|██████▎                 | 261/1000 [04:44<13:07,  1.07s/it]
Epoch Progress:  26%|██████▎                 | 262/1000 [04:45<13:07,  1.07s/it]
Epoch Progress:  26%|██████▎                 | 263/1000 [04:46<13:07,  1.07s/it]
Epoch Progress:  26%|██████▎                 | 264/1000 [04:47<13:06,  1.07s/it]
Epoch Progress:  26%|██████▎                 | 265/1000 [04:48<13:05,  1.07s/it]
Epoch Progress:  27%|██████▍                 | 266/1000 [04:50<13:07,  1.07s/it]
Epoch Progress:  27%|██████▍                 | 267/1000 [04:51<13:13,  1.08s/it]
Epoch Progress:  27%|██████▍                 | 268/1000 [04:52<13:09,  1.08s/it]
Epoch Progress:  27%|██████▍                 | 269/1000 [04:53<13:05,  1.07s/it]
Epoch Progress:  27%|██████▍                 | 270/1000 [04:54<13:02,  1.07s/it]Dataset Size: 1%, Epoch: 271, Train Loss: 2.761599540710449, Val Loss: 9.50660637097481

Epoch Progress:  27%|██████▌                 | 271/1000 [04:55<13:00,  1.07s/it]
Epoch Progress:  27%|██████▌                 | 272/1000 [04:56<12:56,  1.07s/it]
Epoch Progress:  27%|██████▌                 | 273/1000 [04:57<12:54,  1.07s/it]
Epoch Progress:  27%|██████▌                 | 274/1000 [04:58<12:51,  1.06s/it]
Epoch Progress:  28%|██████▌                 | 275/1000 [04:59<12:50,  1.06s/it]
Epoch Progress:  28%|██████▌                 | 276/1000 [05:00<12:51,  1.07s/it]
Epoch Progress:  28%|██████▋                 | 277/1000 [05:01<12:48,  1.06s/it]
Epoch Progress:  28%|██████▋                 | 278/1000 [05:02<12:48,  1.06s/it]
Epoch Progress:  28%|██████▋                 | 279/1000 [05:03<12:46,  1.06s/it]
Epoch Progress:  28%|██████▋                 | 280/1000 [05:04<12:43,  1.06s/it]Dataset Size: 1%, Epoch: 281, Train Loss: 2.6640536785125732, Val Loss: 9.618120706998385

Epoch Progress:  28%|██████▋                 | 281/1000 [05:05<12:42,  1.06s/it]
Epoch Progress:  28%|██████▊                 | 282/1000 [05:07<12:40,  1.06s/it]
Epoch Progress:  28%|██████▊                 | 283/1000 [05:08<12:39,  1.06s/it]
Epoch Progress:  28%|██████▊                 | 284/1000 [05:09<12:38,  1.06s/it]
Epoch Progress:  28%|██████▊                 | 285/1000 [05:10<12:38,  1.06s/it]
Epoch Progress:  29%|██████▊                 | 286/1000 [05:11<12:38,  1.06s/it]
Epoch Progress:  29%|██████▉                 | 287/1000 [05:12<12:36,  1.06s/it]
Epoch Progress:  29%|██████▉                 | 288/1000 [05:13<12:35,  1.06s/it]
Epoch Progress:  29%|██████▉                 | 289/1000 [05:14<12:33,  1.06s/it]
Epoch Progress:  29%|██████▉                 | 290/1000 [05:15<12:30,  1.06s/it]Dataset Size: 1%, Epoch: 291, Train Loss: 2.589254677295685, Val Loss: 9.737570395836464

Epoch Progress:  29%|██████▉                 | 291/1000 [05:16<12:28,  1.06s/it]
Epoch Progress:  29%|███████                 | 292/1000 [05:17<12:27,  1.06s/it]
Epoch Progress:  29%|███████                 | 293/1000 [05:18<12:26,  1.06s/it]
Epoch Progress:  29%|███████                 | 294/1000 [05:19<12:27,  1.06s/it]
Epoch Progress:  30%|███████                 | 295/1000 [05:20<12:28,  1.06s/it]
Epoch Progress:  30%|███████                 | 296/1000 [05:21<12:27,  1.06s/it]
Epoch Progress:  30%|███████▏                | 297/1000 [05:22<12:29,  1.07s/it]
Epoch Progress:  30%|███████▏                | 298/1000 [05:24<12:28,  1.07s/it]
Epoch Progress:  30%|███████▏                | 299/1000 [05:25<12:25,  1.06s/it]
Epoch Progress:  30%|███████▏                | 300/1000 [05:26<12:22,  1.06s/it]Dataset Size: 1%, Epoch: 301, Train Loss: 2.521758735179901, Val Loss: 9.823241429451185

Epoch Progress:  30%|███████▏                | 301/1000 [05:27<12:19,  1.06s/it]
Epoch Progress:  30%|███████▏                | 302/1000 [05:28<12:17,  1.06s/it]
Epoch Progress:  30%|███████▎                | 303/1000 [05:29<12:17,  1.06s/it]
Epoch Progress:  30%|███████▎                | 304/1000 [05:30<12:18,  1.06s/it]
Epoch Progress:  30%|███████▎                | 305/1000 [05:31<12:17,  1.06s/it]
Epoch Progress:  31%|███████▎                | 306/1000 [05:32<12:15,  1.06s/it]
Epoch Progress:  31%|███████▎                | 307/1000 [05:33<12:13,  1.06s/it]
Epoch Progress:  31%|███████▍                | 308/1000 [05:34<12:13,  1.06s/it]
Epoch Progress:  31%|███████▍                | 309/1000 [05:35<12:11,  1.06s/it]
Epoch Progress:  31%|███████▍                | 310/1000 [05:36<12:08,  1.06s/it]Dataset Size: 1%, Epoch: 311, Train Loss: 2.443928301334381, Val Loss: 9.934937208126753

Epoch Progress:  31%|███████▍                | 311/1000 [05:37<12:06,  1.06s/it]
Epoch Progress:  31%|███████▍                | 312/1000 [05:38<12:05,  1.05s/it]
Epoch Progress:  31%|███████▌                | 313/1000 [05:39<12:06,  1.06s/it]
Epoch Progress:  31%|███████▌                | 314/1000 [05:40<12:05,  1.06s/it]
Epoch Progress:  32%|███████▌                | 315/1000 [05:42<12:04,  1.06s/it]
Epoch Progress:  32%|███████▌                | 316/1000 [05:43<12:01,  1.06s/it]
Epoch Progress:  32%|███████▌                | 317/1000 [05:44<12:01,  1.06s/it]
Epoch Progress:  32%|███████▋                | 318/1000 [05:45<11:59,  1.05s/it]
Epoch Progress:  32%|███████▋                | 319/1000 [05:46<11:57,  1.05s/it]
Epoch Progress:  32%|███████▋                | 320/1000 [05:47<11:57,  1.05s/it]Dataset Size: 1%, Epoch: 321, Train Loss: 2.384258508682251, Val Loss: 10.039566553556002

Epoch Progress:  32%|███████▋                | 321/1000 [05:48<11:56,  1.06s/it]
Epoch Progress:  32%|███████▋                | 322/1000 [05:49<11:55,  1.06s/it]
Epoch Progress:  32%|███████▊                | 323/1000 [05:50<11:56,  1.06s/it]
Epoch Progress:  32%|███████▊                | 324/1000 [05:51<11:55,  1.06s/it]
Epoch Progress:  32%|███████▊                | 325/1000 [05:52<11:54,  1.06s/it]
Epoch Progress:  33%|███████▊                | 326/1000 [05:53<11:52,  1.06s/it]
Epoch Progress:  33%|███████▊                | 327/1000 [05:54<11:54,  1.06s/it]
Epoch Progress:  33%|███████▊                | 328/1000 [05:55<11:51,  1.06s/it]
Epoch Progress:  33%|███████▉                | 329/1000 [05:56<11:50,  1.06s/it]
Epoch Progress:  33%|███████▉                | 330/1000 [05:57<11:47,  1.06s/it]Dataset Size: 1%, Epoch: 331, Train Loss: 2.331229507923126, Val Loss: 10.130320622370792

Epoch Progress:  33%|███████▉                | 331/1000 [05:58<11:46,  1.06s/it]
Epoch Progress:  33%|███████▉                | 332/1000 [05:59<11:47,  1.06s/it]
Epoch Progress:  33%|███████▉                | 333/1000 [06:01<11:45,  1.06s/it]
Epoch Progress:  33%|████████                | 334/1000 [06:02<11:43,  1.06s/it]
Epoch Progress:  34%|████████                | 335/1000 [06:03<11:42,  1.06s/it]
Epoch Progress:  34%|████████                | 336/1000 [06:04<11:41,  1.06s/it]
Epoch Progress:  34%|████████                | 337/1000 [06:05<11:40,  1.06s/it]
Epoch Progress:  34%|████████                | 338/1000 [06:06<11:38,  1.05s/it]
Epoch Progress:  34%|████████▏               | 339/1000 [06:07<11:36,  1.05s/it]
Epoch Progress:  34%|████████▏               | 340/1000 [06:08<11:35,  1.05s/it]Dataset Size: 1%, Epoch: 341, Train Loss: 2.2787447571754456, Val Loss: 10.228827476501465

Epoch Progress:  34%|████████▏               | 341/1000 [06:09<11:34,  1.05s/it]
Epoch Progress:  34%|████████▏               | 342/1000 [06:10<11:35,  1.06s/it]
Epoch Progress:  34%|████████▏               | 343/1000 [06:11<11:34,  1.06s/it]
Epoch Progress:  34%|████████▎               | 344/1000 [06:12<11:33,  1.06s/it]
Epoch Progress:  34%|████████▎               | 345/1000 [06:13<11:32,  1.06s/it]
Epoch Progress:  35%|████████▎               | 346/1000 [06:14<11:31,  1.06s/it]
Epoch Progress:  35%|████████▎               | 347/1000 [06:15<11:29,  1.06s/it]
Epoch Progress:  35%|████████▎               | 348/1000 [06:16<11:28,  1.06s/it]
Epoch Progress:  35%|████████▍               | 349/1000 [06:17<11:27,  1.06s/it]
Epoch Progress:  35%|████████▍               | 350/1000 [06:18<11:25,  1.05s/it]Dataset Size: 1%, Epoch: 351, Train Loss: 2.2261979579925537, Val Loss: 10.328415895119692

Epoch Progress:  35%|████████▍               | 351/1000 [06:20<11:27,  1.06s/it]
Epoch Progress:  35%|████████▍               | 352/1000 [06:21<11:25,  1.06s/it]
Epoch Progress:  35%|████████▍               | 353/1000 [06:22<11:24,  1.06s/it]
Epoch Progress:  35%|████████▍               | 354/1000 [06:23<11:22,  1.06s/it]
Epoch Progress:  36%|████████▌               | 355/1000 [06:24<11:23,  1.06s/it]
Epoch Progress:  36%|████████▌               | 356/1000 [06:25<11:24,  1.06s/it]
Epoch Progress:  36%|████████▌               | 357/1000 [06:26<11:25,  1.07s/it]
Epoch Progress:  36%|████████▌               | 358/1000 [06:27<11:22,  1.06s/it]
Epoch Progress:  36%|████████▌               | 359/1000 [06:28<11:20,  1.06s/it]
Epoch Progress:  36%|████████▋               | 360/1000 [06:29<11:18,  1.06s/it]Dataset Size: 1%, Epoch: 361, Train Loss: 2.162055730819702, Val Loss: 10.403458913167318

Epoch Progress:  36%|████████▋               | 361/1000 [06:30<11:18,  1.06s/it]
Epoch Progress:  36%|████████▋               | 362/1000 [06:31<11:16,  1.06s/it]
Epoch Progress:  36%|████████▋               | 363/1000 [06:32<11:14,  1.06s/it]
Epoch Progress:  36%|████████▋               | 364/1000 [06:33<11:12,  1.06s/it]
Epoch Progress:  36%|████████▊               | 365/1000 [06:34<11:11,  1.06s/it]
Epoch Progress:  37%|████████▊               | 366/1000 [06:35<11:08,  1.05s/it]
Epoch Progress:  37%|████████▊               | 367/1000 [06:36<11:07,  1.05s/it]
Epoch Progress:  37%|████████▊               | 368/1000 [06:38<11:06,  1.05s/it]
Epoch Progress:  37%|████████▊               | 369/1000 [06:39<11:05,  1.05s/it]
Epoch Progress:  37%|████████▉               | 370/1000 [06:40<11:05,  1.06s/it]Dataset Size: 1%, Epoch: 371, Train Loss: 2.1399739384651184, Val Loss: 10.496765087812374

Epoch Progress:  37%|████████▉               | 371/1000 [06:41<11:04,  1.06s/it]
Epoch Progress:  37%|████████▉               | 372/1000 [06:42<11:02,  1.06s/it]
Epoch Progress:  37%|████████▉               | 373/1000 [06:43<11:01,  1.06s/it]
Epoch Progress:  37%|████████▉               | 374/1000 [06:44<10:59,  1.05s/it]
Epoch Progress:  38%|█████████               | 375/1000 [06:45<10:58,  1.05s/it]
Epoch Progress:  38%|█████████               | 376/1000 [06:46<10:56,  1.05s/it]
Epoch Progress:  38%|█████████               | 377/1000 [06:47<10:55,  1.05s/it]
Epoch Progress:  38%|█████████               | 378/1000 [06:48<10:55,  1.05s/it]
Epoch Progress:  38%|█████████               | 379/1000 [06:49<10:54,  1.05s/it]
Epoch Progress:  38%|█████████               | 380/1000 [06:50<10:54,  1.06s/it]Dataset Size: 1%, Epoch: 381, Train Loss: 2.089624345302582, Val Loss: 10.592494940146421

Epoch Progress:  38%|█████████▏              | 381/1000 [06:51<10:54,  1.06s/it]
Epoch Progress:  38%|█████████▏              | 382/1000 [06:52<10:52,  1.06s/it]
Epoch Progress:  38%|█████████▏              | 383/1000 [06:53<10:51,  1.06s/it]
Epoch Progress:  38%|█████████▏              | 384/1000 [06:54<10:49,  1.05s/it]
Epoch Progress:  38%|█████████▏              | 385/1000 [06:55<10:47,  1.05s/it]
Epoch Progress:  39%|█████████▎              | 386/1000 [06:57<10:46,  1.05s/it]
Epoch Progress:  39%|█████████▎              | 387/1000 [06:58<10:48,  1.06s/it]
Epoch Progress:  39%|█████████▎              | 388/1000 [06:59<10:47,  1.06s/it]
Epoch Progress:  39%|█████████▎              | 389/1000 [07:00<10:47,  1.06s/it]
Epoch Progress:  39%|█████████▎              | 390/1000 [07:01<10:46,  1.06s/it]Dataset Size: 1%, Epoch: 391, Train Loss: 2.064455896615982, Val Loss: 10.661152399503267

Epoch Progress:  39%|█████████▍              | 391/1000 [07:02<10:44,  1.06s/it]
Epoch Progress:  39%|█████████▍              | 392/1000 [07:03<10:42,  1.06s/it]
Epoch Progress:  39%|█████████▍              | 393/1000 [07:04<10:40,  1.06s/it]
Epoch Progress:  39%|█████████▍              | 394/1000 [07:05<10:39,  1.05s/it]
Epoch Progress:  40%|█████████▍              | 395/1000 [07:06<10:38,  1.06s/it]
Epoch Progress:  40%|█████████▌              | 396/1000 [07:07<10:37,  1.06s/it]
Epoch Progress:  40%|█████████▌              | 397/1000 [07:08<10:35,  1.05s/it]
Epoch Progress:  40%|█████████▌              | 398/1000 [07:09<10:35,  1.05s/it]
Epoch Progress:  40%|█████████▌              | 399/1000 [07:10<10:36,  1.06s/it]
Epoch Progress:  40%|█████████▌              | 400/1000 [07:11<10:34,  1.06s/it]Dataset Size: 1%, Epoch: 401, Train Loss: 1.998457431793213, Val Loss: 10.746669402489296

Epoch Progress:  40%|█████████▌              | 401/1000 [07:12<10:32,  1.06s/it]
Epoch Progress:  40%|█████████▋              | 402/1000 [07:13<10:31,  1.06s/it]
Epoch Progress:  40%|█████████▋              | 403/1000 [07:14<10:29,  1.05s/it]
Epoch Progress:  40%|█████████▋              | 404/1000 [07:16<10:27,  1.05s/it]
Epoch Progress:  40%|█████████▋              | 405/1000 [07:17<10:26,  1.05s/it]
Epoch Progress:  41%|█████████▋              | 406/1000 [07:18<10:25,  1.05s/it]
Epoch Progress:  41%|█████████▊              | 407/1000 [07:19<10:25,  1.05s/it]
Epoch Progress:  41%|█████████▊              | 408/1000 [07:20<10:25,  1.06s/it]
Epoch Progress:  41%|█████████▊              | 409/1000 [07:21<10:24,  1.06s/it]
Epoch Progress:  41%|█████████▊              | 410/1000 [07:22<10:22,  1.05s/it]Dataset Size: 1%, Epoch: 411, Train Loss: 1.9957781732082367, Val Loss: 10.819293511219513

Epoch Progress:  41%|█████████▊              | 411/1000 [07:23<10:21,  1.06s/it]
Epoch Progress:  41%|█████████▉              | 412/1000 [07:24<10:20,  1.06s/it]
Epoch Progress:  41%|█████████▉              | 413/1000 [07:25<10:19,  1.06s/it]
Epoch Progress:  41%|█████████▉              | 414/1000 [07:26<10:18,  1.06s/it]
Epoch Progress:  42%|█████████▉              | 415/1000 [07:27<10:17,  1.06s/it]
Epoch Progress:  42%|█████████▉              | 416/1000 [07:28<10:18,  1.06s/it]
Epoch Progress:  42%|██████████              | 417/1000 [07:29<10:21,  1.07s/it]
Epoch Progress:  42%|██████████              | 418/1000 [07:30<10:20,  1.07s/it]
Epoch Progress:  42%|██████████              | 419/1000 [07:31<10:17,  1.06s/it]
Epoch Progress:  42%|██████████              | 420/1000 [07:32<10:16,  1.06s/it]Dataset Size: 1%, Epoch: 421, Train Loss: 1.9602761268615723, Val Loss: 10.903883542769995

Epoch Progress:  42%|██████████              | 421/1000 [07:34<10:15,  1.06s/it]
Epoch Progress:  42%|██████████▏             | 422/1000 [07:35<10:13,  1.06s/it]
Epoch Progress:  42%|██████████▏             | 423/1000 [07:36<10:11,  1.06s/it]
Epoch Progress:  42%|██████████▏             | 424/1000 [07:37<10:09,  1.06s/it]
Epoch Progress:  42%|██████████▏             | 425/1000 [07:38<10:08,  1.06s/it]
Epoch Progress:  43%|██████████▏             | 426/1000 [07:39<10:07,  1.06s/it]
Epoch Progress:  43%|██████████▏             | 427/1000 [07:40<10:08,  1.06s/it]
Epoch Progress:  43%|██████████▎             | 428/1000 [07:41<10:07,  1.06s/it]
Epoch Progress:  43%|██████████▎             | 429/1000 [07:42<10:07,  1.06s/it]
Epoch Progress:  43%|██████████▎             | 430/1000 [07:43<10:04,  1.06s/it]Dataset Size: 1%, Epoch: 431, Train Loss: 1.9344433844089508, Val Loss: 10.982154259314903

Epoch Progress:  43%|██████████▎             | 431/1000 [07:44<10:02,  1.06s/it]
Epoch Progress:  43%|██████████▎             | 432/1000 [07:45<10:03,  1.06s/it]
Epoch Progress:  43%|██████████▍             | 433/1000 [07:46<10:01,  1.06s/it]
Epoch Progress:  43%|██████████▍             | 434/1000 [07:47<09:59,  1.06s/it]
Epoch Progress:  44%|██████████▍             | 435/1000 [07:48<09:57,  1.06s/it]
Epoch Progress:  44%|██████████▍             | 436/1000 [07:49<09:57,  1.06s/it]
Epoch Progress:  44%|██████████▍             | 437/1000 [07:50<09:55,  1.06s/it]
Epoch Progress:  44%|██████████▌             | 438/1000 [07:52<09:54,  1.06s/it]
Epoch Progress:  44%|██████████▌             | 439/1000 [07:53<09:53,  1.06s/it]
Epoch Progress:  44%|██████████▌             | 440/1000 [07:54<09:51,  1.06s/it]Dataset Size: 1%, Epoch: 441, Train Loss: 1.9127311408519745, Val Loss: 11.044229531899477

Epoch Progress:  44%|██████████▌             | 441/1000 [07:55<09:49,  1.05s/it]
Epoch Progress:  44%|██████████▌             | 442/1000 [07:56<09:47,  1.05s/it]
Epoch Progress:  44%|██████████▋             | 443/1000 [07:57<09:46,  1.05s/it]
Epoch Progress:  44%|██████████▋             | 444/1000 [07:58<09:45,  1.05s/it]
Epoch Progress:  44%|██████████▋             | 445/1000 [07:59<09:44,  1.05s/it]
Epoch Progress:  45%|██████████▋             | 446/1000 [08:00<09:48,  1.06s/it]
Epoch Progress:  45%|██████████▋             | 447/1000 [08:01<09:47,  1.06s/it]
Epoch Progress:  45%|██████████▊             | 448/1000 [08:02<09:45,  1.06s/it]
Epoch Progress:  45%|██████████▊             | 449/1000 [08:03<09:43,  1.06s/it]
Epoch Progress:  45%|██████████▊             | 450/1000 [08:04<09:41,  1.06s/it]Dataset Size: 1%, Epoch: 451, Train Loss: 1.8874320089817047, Val Loss: 11.102358451256386

Epoch Progress:  45%|██████████▊             | 451/1000 [08:05<09:39,  1.06s/it]
Epoch Progress:  45%|██████████▊             | 452/1000 [08:06<09:38,  1.06s/it]
Epoch Progress:  45%|██████████▊             | 453/1000 [08:07<09:36,  1.05s/it]
Epoch Progress:  45%|██████████▉             | 454/1000 [08:08<09:36,  1.06s/it]
Epoch Progress:  46%|██████████▉             | 455/1000 [08:10<09:36,  1.06s/it]
Epoch Progress:  46%|██████████▉             | 456/1000 [08:11<09:35,  1.06s/it]
Epoch Progress:  46%|██████████▉             | 457/1000 [08:12<09:33,  1.06s/it]
Epoch Progress:  46%|██████████▉             | 458/1000 [08:13<09:32,  1.06s/it]
Epoch Progress:  46%|███████████             | 459/1000 [08:14<09:31,  1.06s/it]
Epoch Progress:  46%|███████████             | 460/1000 [08:15<09:30,  1.06s/it]Dataset Size: 1%, Epoch: 461, Train Loss: 1.8631418943405151, Val Loss: 11.166738559038212

Epoch Progress:  46%|███████████             | 461/1000 [08:16<09:28,  1.05s/it]
Epoch Progress:  46%|███████████             | 462/1000 [08:17<09:27,  1.06s/it]
Epoch Progress:  46%|███████████             | 463/1000 [08:18<09:26,  1.05s/it]
Epoch Progress:  46%|███████████▏            | 464/1000 [08:19<09:24,  1.05s/it]
Epoch Progress:  46%|███████████▏            | 465/1000 [08:20<09:25,  1.06s/it]
Epoch Progress:  47%|███████████▏            | 466/1000 [08:21<09:24,  1.06s/it]
Epoch Progress:  47%|███████████▏            | 467/1000 [08:22<09:22,  1.06s/it]
Epoch Progress:  47%|███████████▏            | 468/1000 [08:23<09:21,  1.06s/it]
Epoch Progress:  47%|███████████▎            | 469/1000 [08:24<09:20,  1.05s/it]
Epoch Progress:  47%|███████████▎            | 470/1000 [08:25<09:18,  1.05s/it]Dataset Size: 1%, Epoch: 471, Train Loss: 1.841493159532547, Val Loss: 11.238615916325497

Epoch Progress:  47%|███████████▎            | 471/1000 [08:26<09:18,  1.06s/it]
Epoch Progress:  47%|███████████▎            | 472/1000 [08:27<09:16,  1.05s/it]
Epoch Progress:  47%|███████████▎            | 473/1000 [08:29<09:16,  1.06s/it]
Epoch Progress:  47%|███████████▍            | 474/1000 [08:30<09:16,  1.06s/it]
Epoch Progress:  48%|███████████▍            | 475/1000 [08:31<09:15,  1.06s/it]
Epoch Progress:  48%|███████████▍            | 476/1000 [08:32<09:15,  1.06s/it]
Epoch Progress:  48%|███████████▍            | 477/1000 [08:33<09:15,  1.06s/it]
Epoch Progress:  48%|███████████▍            | 478/1000 [08:34<09:13,  1.06s/it]
Epoch Progress:  48%|███████████▍            | 479/1000 [08:35<09:11,  1.06s/it]
Epoch Progress:  48%|███████████▌            | 480/1000 [08:36<09:09,  1.06s/it]Dataset Size: 1%, Epoch: 481, Train Loss: 1.8204830586910248, Val Loss: 11.303433247101612

Epoch Progress:  48%|███████████▌            | 481/1000 [08:37<09:08,  1.06s/it]
Epoch Progress:  48%|███████████▌            | 482/1000 [08:38<09:07,  1.06s/it]
Epoch Progress:  48%|███████████▌            | 483/1000 [08:39<09:06,  1.06s/it]
Epoch Progress:  48%|███████████▌            | 484/1000 [08:40<09:06,  1.06s/it]
Epoch Progress:  48%|███████████▋            | 485/1000 [08:41<09:05,  1.06s/it]
Epoch Progress:  49%|███████████▋            | 486/1000 [08:42<09:04,  1.06s/it]
Epoch Progress:  49%|███████████▋            | 487/1000 [08:43<09:02,  1.06s/it]
Epoch Progress:  49%|███████████▋            | 488/1000 [08:44<09:01,  1.06s/it]
Epoch Progress:  49%|███████████▋            | 489/1000 [08:45<09:00,  1.06s/it]
Epoch Progress:  49%|███████████▊            | 490/1000 [08:46<08:58,  1.06s/it]Dataset Size: 1%, Epoch: 491, Train Loss: 1.7976279854774475, Val Loss: 11.361540011870556

Epoch Progress:  49%|███████████▊            | 491/1000 [08:48<08:57,  1.06s/it]
Epoch Progress:  49%|███████████▊            | 492/1000 [08:49<08:56,  1.06s/it]
Epoch Progress:  49%|███████████▊            | 493/1000 [08:50<08:56,  1.06s/it]
Epoch Progress:  49%|███████████▊            | 494/1000 [08:51<08:55,  1.06s/it]
Epoch Progress:  50%|███████████▉            | 495/1000 [08:52<08:53,  1.06s/it]
Epoch Progress:  50%|███████████▉            | 496/1000 [08:53<08:53,  1.06s/it]
Epoch Progress:  50%|███████████▉            | 497/1000 [08:54<08:51,  1.06s/it]
Epoch Progress:  50%|███████████▉            | 498/1000 [08:55<08:50,  1.06s/it]
Epoch Progress:  50%|███████████▉            | 499/1000 [08:56<08:49,  1.06s/it]
Epoch Progress:  50%|████████████            | 500/1000 [08:57<08:47,  1.06s/it]Dataset Size: 1%, Epoch: 501, Train Loss: 1.784915179014206, Val Loss: 11.408896030523838

Epoch Progress:  50%|████████████            | 501/1000 [08:58<08:46,  1.06s/it]
Epoch Progress:  50%|████████████            | 502/1000 [08:59<08:46,  1.06s/it]
Epoch Progress:  50%|████████████            | 503/1000 [09:00<08:46,  1.06s/it]
Epoch Progress:  50%|████████████            | 504/1000 [09:01<08:44,  1.06s/it]
Epoch Progress:  50%|████████████            | 505/1000 [09:02<08:43,  1.06s/it]
Epoch Progress:  51%|████████████▏           | 506/1000 [09:03<08:43,  1.06s/it]
Epoch Progress:  51%|████████████▏           | 507/1000 [09:04<08:43,  1.06s/it]
Epoch Progress:  51%|████████████▏           | 508/1000 [09:06<08:42,  1.06s/it]
Epoch Progress:  51%|████████████▏           | 509/1000 [09:07<08:40,  1.06s/it]
Epoch Progress:  51%|████████████▏           | 510/1000 [09:08<08:39,  1.06s/it]Dataset Size: 1%, Epoch: 511, Train Loss: 1.764525830745697, Val Loss: 11.47072809170454

Epoch Progress:  51%|████████████▎           | 511/1000 [09:09<08:37,  1.06s/it]
Epoch Progress:  51%|████████████▎           | 512/1000 [09:10<08:36,  1.06s/it]
Epoch Progress:  51%|████████████▎           | 513/1000 [09:11<08:34,  1.06s/it]
Epoch Progress:  51%|████████████▎           | 514/1000 [09:12<08:33,  1.06s/it]
Epoch Progress:  52%|████████████▎           | 515/1000 [09:13<08:31,  1.06s/it]
Epoch Progress:  52%|████████████▍           | 516/1000 [09:14<08:31,  1.06s/it]
Epoch Progress:  52%|████████████▍           | 517/1000 [09:15<08:29,  1.06s/it]
Epoch Progress:  52%|████████████▍           | 518/1000 [09:16<08:28,  1.06s/it]
Epoch Progress:  52%|████████████▍           | 519/1000 [09:17<08:27,  1.06s/it]
Epoch Progress:  52%|████████████▍           | 520/1000 [09:18<08:26,  1.06s/it]Dataset Size: 1%, Epoch: 521, Train Loss: 1.748254656791687, Val Loss: 11.534647672604292

Epoch Progress:  52%|████████████▌           | 521/1000 [09:19<08:25,  1.06s/it]
Epoch Progress:  52%|████████████▌           | 522/1000 [09:20<08:25,  1.06s/it]
Epoch Progress:  52%|████████████▌           | 523/1000 [09:21<08:23,  1.06s/it]
Epoch Progress:  52%|████████████▌           | 524/1000 [09:22<08:22,  1.06s/it]
Epoch Progress:  52%|████████████▌           | 525/1000 [09:23<08:21,  1.06s/it]
Epoch Progress:  53%|████████████▌           | 526/1000 [09:25<08:20,  1.06s/it]
Epoch Progress:  53%|████████████▋           | 527/1000 [09:26<08:19,  1.06s/it]
Epoch Progress:  53%|████████████▋           | 528/1000 [09:27<08:18,  1.06s/it]
Epoch Progress:  53%|████████████▋           | 529/1000 [09:28<08:17,  1.06s/it]
Epoch Progress:  53%|████████████▋           | 530/1000 [09:29<08:16,  1.06s/it]Dataset Size: 1%, Epoch: 531, Train Loss: 1.728655457496643, Val Loss: 11.58367350162604

Epoch Progress:  53%|████████████▋           | 531/1000 [09:30<08:16,  1.06s/it]
Epoch Progress:  53%|████████████▊           | 532/1000 [09:31<08:14,  1.06s/it]
Epoch Progress:  53%|████████████▊           | 533/1000 [09:32<08:13,  1.06s/it]
Epoch Progress:  53%|████████████▊           | 534/1000 [09:33<08:11,  1.06s/it]
Epoch Progress:  54%|████████████▊           | 535/1000 [09:34<08:10,  1.05s/it]
Epoch Progress:  54%|████████████▊           | 536/1000 [09:35<08:11,  1.06s/it]
Epoch Progress:  54%|████████████▉           | 537/1000 [09:36<08:10,  1.06s/it]
Epoch Progress:  54%|████████████▉           | 538/1000 [09:37<08:08,  1.06s/it]
Epoch Progress:  54%|████████████▉           | 539/1000 [09:38<08:06,  1.06s/it]
Epoch Progress:  54%|████████████▉           | 540/1000 [09:39<08:06,  1.06s/it]Dataset Size: 1%, Epoch: 541, Train Loss: 1.7263100445270538, Val Loss: 11.625894570961977

Epoch Progress:  54%|████████████▉           | 541/1000 [09:40<08:04,  1.06s/it]
Epoch Progress:  54%|█████████████           | 542/1000 [09:41<08:03,  1.06s/it]
Epoch Progress:  54%|█████████████           | 543/1000 [09:43<08:02,  1.06s/it]
Epoch Progress:  54%|█████████████           | 544/1000 [09:44<08:01,  1.06s/it]
Epoch Progress:  55%|█████████████           | 545/1000 [09:45<08:00,  1.06s/it]
Epoch Progress:  55%|█████████████           | 546/1000 [09:46<07:59,  1.06s/it]
Epoch Progress:  55%|█████████████▏          | 547/1000 [09:47<07:58,  1.06s/it]
Epoch Progress:  55%|█████████████▏          | 548/1000 [09:48<07:57,  1.06s/it]
Epoch Progress:  55%|█████████████▏          | 549/1000 [09:49<07:56,  1.06s/it]
Epoch Progress:  55%|█████████████▏          | 550/1000 [09:50<07:55,  1.06s/it]Dataset Size: 1%, Epoch: 551, Train Loss: 1.7044876515865326, Val Loss: 11.68765456859882

Epoch Progress:  55%|█████████████▏          | 551/1000 [09:51<07:54,  1.06s/it]
Epoch Progress:  55%|█████████████▏          | 552/1000 [09:52<07:53,  1.06s/it]
Epoch Progress:  55%|█████████████▎          | 553/1000 [09:53<07:51,  1.06s/it]
Epoch Progress:  55%|█████████████▎          | 554/1000 [09:54<07:50,  1.06s/it]
Epoch Progress:  56%|█████████████▎          | 555/1000 [09:55<07:49,  1.05s/it]
Epoch Progress:  56%|█████████████▎          | 556/1000 [09:56<07:48,  1.06s/it]
Epoch Progress:  56%|█████████████▎          | 557/1000 [09:57<07:47,  1.06s/it]
Epoch Progress:  56%|█████████████▍          | 558/1000 [09:58<07:46,  1.05s/it]
Epoch Progress:  56%|█████████████▍          | 559/1000 [09:59<07:46,  1.06s/it]
Epoch Progress:  56%|█████████████▍          | 560/1000 [10:00<07:44,  1.06s/it]Dataset Size: 1%, Epoch: 561, Train Loss: 1.692302167415619, Val Loss: 11.733989373231546

Epoch Progress:  56%|█████████████▍          | 561/1000 [10:02<07:43,  1.06s/it]
Epoch Progress:  56%|█████████████▍          | 562/1000 [10:03<07:42,  1.06s/it]
Epoch Progress:  56%|█████████████▌          | 563/1000 [10:04<07:41,  1.06s/it]
Epoch Progress:  56%|█████████████▌          | 564/1000 [10:05<07:40,  1.06s/it]
Epoch Progress:  56%|█████████████▌          | 565/1000 [10:06<07:39,  1.06s/it]
Epoch Progress:  57%|█████████████▌          | 566/1000 [10:07<07:39,  1.06s/it]
Epoch Progress:  57%|█████████████▌          | 567/1000 [10:08<07:39,  1.06s/it]
Epoch Progress:  57%|█████████████▋          | 568/1000 [10:09<07:37,  1.06s/it]
Epoch Progress:  57%|█████████████▋          | 569/1000 [10:10<07:36,  1.06s/it]
Epoch Progress:  57%|█████████████▋          | 570/1000 [10:11<07:35,  1.06s/it]Dataset Size: 1%, Epoch: 571, Train Loss: 1.6796038746833801, Val Loss: 11.769542107215294

Epoch Progress:  57%|█████████████▋          | 571/1000 [10:12<07:33,  1.06s/it]
Epoch Progress:  57%|█████████████▋          | 572/1000 [10:13<07:32,  1.06s/it]
Epoch Progress:  57%|█████████████▊          | 573/1000 [10:14<07:31,  1.06s/it]
Epoch Progress:  57%|█████████████▊          | 574/1000 [10:15<07:31,  1.06s/it]
Epoch Progress:  57%|█████████████▊          | 575/1000 [10:16<07:30,  1.06s/it]
Epoch Progress:  58%|█████████████▊          | 576/1000 [10:17<07:29,  1.06s/it]
Epoch Progress:  58%|█████████████▊          | 577/1000 [10:18<07:28,  1.06s/it]
Epoch Progress:  58%|█████████████▊          | 578/1000 [10:20<07:27,  1.06s/it]
Epoch Progress:  58%|█████████████▉          | 579/1000 [10:21<07:27,  1.06s/it]
Epoch Progress:  58%|█████████████▉          | 580/1000 [10:22<07:25,  1.06s/it]Dataset Size: 1%, Epoch: 581, Train Loss: 1.6626669764518738, Val Loss: 11.824969365046574

Epoch Progress:  58%|█████████████▉          | 581/1000 [10:23<07:24,  1.06s/it]
Epoch Progress:  58%|█████████████▉          | 582/1000 [10:24<07:23,  1.06s/it]
Epoch Progress:  58%|█████████████▉          | 583/1000 [10:25<07:22,  1.06s/it]
Epoch Progress:  58%|██████████████          | 584/1000 [10:26<07:21,  1.06s/it]
Epoch Progress:  58%|██████████████          | 585/1000 [10:27<07:20,  1.06s/it]
Epoch Progress:  59%|██████████████          | 586/1000 [10:28<07:19,  1.06s/it]
Epoch Progress:  59%|██████████████          | 587/1000 [10:29<07:18,  1.06s/it]
Epoch Progress:  59%|██████████████          | 588/1000 [10:30<07:18,  1.06s/it]
Epoch Progress:  59%|██████████████▏         | 589/1000 [10:31<07:16,  1.06s/it]
Epoch Progress:  59%|██████████████▏         | 590/1000 [10:32<07:14,  1.06s/it]Dataset Size: 1%, Epoch: 591, Train Loss: 1.660144954919815, Val Loss: 11.852852968069223

Epoch Progress:  59%|██████████████▏         | 591/1000 [10:33<07:12,  1.06s/it]
Epoch Progress:  59%|██████████████▏         | 592/1000 [10:34<07:10,  1.06s/it]
Epoch Progress:  59%|██████████████▏         | 593/1000 [10:35<07:10,  1.06s/it]
Epoch Progress:  59%|██████████████▎         | 594/1000 [10:36<07:08,  1.06s/it]
Epoch Progress:  60%|██████████████▎         | 595/1000 [10:38<07:07,  1.05s/it]
Epoch Progress:  60%|██████████████▎         | 596/1000 [10:39<07:08,  1.06s/it]
Epoch Progress:  60%|██████████████▎         | 597/1000 [10:40<07:11,  1.07s/it]
Epoch Progress:  60%|██████████████▎         | 598/1000 [10:41<07:09,  1.07s/it]
Epoch Progress:  60%|██████████████▍         | 599/1000 [10:42<07:06,  1.06s/it]
Epoch Progress:  60%|██████████████▍         | 600/1000 [10:43<07:04,  1.06s/it]Dataset Size: 1%, Epoch: 601, Train Loss: 1.6434841752052307, Val Loss: 11.90434944935334

Epoch Progress:  60%|██████████████▍         | 601/1000 [10:44<07:03,  1.06s/it]
Epoch Progress:  60%|██████████████▍         | 602/1000 [10:45<07:01,  1.06s/it]
Epoch Progress:  60%|██████████████▍         | 603/1000 [10:46<07:00,  1.06s/it]
Epoch Progress:  60%|██████████████▍         | 604/1000 [10:47<06:58,  1.06s/it]
Epoch Progress:  60%|██████████████▌         | 605/1000 [10:48<06:57,  1.06s/it]
Epoch Progress:  61%|██████████████▌         | 606/1000 [10:49<06:56,  1.06s/it]
Epoch Progress:  61%|██████████████▌         | 607/1000 [10:50<06:56,  1.06s/it]
Epoch Progress:  61%|██████████████▌         | 608/1000 [10:51<06:54,  1.06s/it]
Epoch Progress:  61%|██████████████▌         | 609/1000 [10:52<06:53,  1.06s/it]
Epoch Progress:  61%|██████████████▋         | 610/1000 [10:53<06:52,  1.06s/it]Dataset Size: 1%, Epoch: 611, Train Loss: 1.637639343738556, Val Loss: 11.937233582521097

Epoch Progress:  61%|██████████████▋         | 611/1000 [10:54<06:51,  1.06s/it]
Epoch Progress:  61%|██████████████▋         | 612/1000 [10:56<06:49,  1.06s/it]
Epoch Progress:  61%|██████████████▋         | 613/1000 [10:57<06:48,  1.06s/it]
Epoch Progress:  61%|██████████████▋         | 614/1000 [10:58<06:46,  1.05s/it]
Epoch Progress:  62%|██████████████▊         | 615/1000 [10:59<06:45,  1.05s/it]
Epoch Progress:  62%|██████████████▊         | 616/1000 [11:00<06:45,  1.06s/it]
Epoch Progress:  62%|██████████████▊         | 617/1000 [11:01<06:44,  1.06s/it]
Epoch Progress:  62%|██████████████▊         | 618/1000 [11:02<06:44,  1.06s/it]
Epoch Progress:  62%|██████████████▊         | 619/1000 [11:03<06:42,  1.06s/it]
Epoch Progress:  62%|██████████████▉         | 620/1000 [11:04<06:40,  1.06s/it]Dataset Size: 1%, Epoch: 621, Train Loss: 1.6242735087871552, Val Loss: 11.984600654015175

Epoch Progress:  62%|██████████████▉         | 621/1000 [11:05<06:39,  1.06s/it]
Epoch Progress:  62%|██████████████▉         | 622/1000 [11:06<06:38,  1.06s/it]
Epoch Progress:  62%|██████████████▉         | 623/1000 [11:07<06:37,  1.05s/it]
Epoch Progress:  62%|██████████████▉         | 624/1000 [11:08<06:36,  1.05s/it]
Epoch Progress:  62%|███████████████         | 625/1000 [11:09<06:36,  1.06s/it]
Epoch Progress:  63%|███████████████         | 626/1000 [11:10<06:36,  1.06s/it]
Epoch Progress:  63%|███████████████         | 627/1000 [11:11<06:36,  1.06s/it]
Epoch Progress:  63%|███████████████         | 628/1000 [11:12<06:34,  1.06s/it]
Epoch Progress:  63%|███████████████         | 629/1000 [11:14<06:32,  1.06s/it]
Epoch Progress:  63%|███████████████         | 630/1000 [11:15<06:31,  1.06s/it]Dataset Size: 1%, Epoch: 631, Train Loss: 1.6256430745124817, Val Loss: 12.021752797640287

Epoch Progress:  63%|███████████████▏        | 631/1000 [11:16<06:29,  1.06s/it]
Epoch Progress:  63%|███████████████▏        | 632/1000 [11:17<06:28,  1.05s/it]
Epoch Progress:  63%|███████████████▏        | 633/1000 [11:18<06:26,  1.05s/it]
Epoch Progress:  63%|███████████████▏        | 634/1000 [11:19<06:25,  1.05s/it]
Epoch Progress:  64%|███████████████▏        | 635/1000 [11:20<06:25,  1.06s/it]
Epoch Progress:  64%|███████████████▎        | 636/1000 [11:21<06:24,  1.06s/it]
Epoch Progress:  64%|███████████████▎        | 637/1000 [11:22<06:22,  1.05s/it]
Epoch Progress:  64%|███████████████▎        | 638/1000 [11:23<06:22,  1.06s/it]
Epoch Progress:  64%|███████████████▎        | 639/1000 [11:24<06:20,  1.06s/it]
Epoch Progress:  64%|███████████████▎        | 640/1000 [11:25<06:20,  1.06s/it]Dataset Size: 1%, Epoch: 641, Train Loss: 1.610226035118103, Val Loss: 12.048863215324205

Epoch Progress:  64%|███████████████▍        | 641/1000 [11:26<06:19,  1.06s/it]
Epoch Progress:  64%|███████████████▍        | 642/1000 [11:27<06:17,  1.05s/it]
Epoch Progress:  64%|███████████████▍        | 643/1000 [11:28<06:16,  1.05s/it]
Epoch Progress:  64%|███████████████▍        | 644/1000 [11:29<06:16,  1.06s/it]
Epoch Progress:  64%|███████████████▍        | 645/1000 [11:30<06:14,  1.06s/it]
Epoch Progress:  65%|███████████████▌        | 646/1000 [11:31<06:13,  1.06s/it]
Epoch Progress:  65%|███████████████▌        | 647/1000 [11:33<06:12,  1.06s/it]
Epoch Progress:  65%|███████████████▌        | 648/1000 [11:34<06:11,  1.05s/it]
Epoch Progress:  65%|███████████████▌        | 649/1000 [11:35<06:10,  1.05s/it]
Epoch Progress:  65%|███████████████▌        | 650/1000 [11:36<06:08,  1.05s/it]Dataset Size: 1%, Epoch: 651, Train Loss: 1.5998525321483612, Val Loss: 12.084255340771797

Epoch Progress:  65%|███████████████▌        | 651/1000 [11:37<06:07,  1.05s/it]
Epoch Progress:  65%|███████████████▋        | 652/1000 [11:38<06:06,  1.05s/it]
Epoch Progress:  65%|███████████████▋        | 653/1000 [11:39<06:04,  1.05s/it]
Epoch Progress:  65%|███████████████▋        | 654/1000 [11:40<06:05,  1.06s/it]
Epoch Progress:  66%|███████████████▋        | 655/1000 [11:41<06:03,  1.05s/it]
Epoch Progress:  66%|███████████████▋        | 656/1000 [11:42<06:03,  1.06s/it]
Epoch Progress:  66%|███████████████▊        | 657/1000 [11:43<06:03,  1.06s/it]
Epoch Progress:  66%|███████████████▊        | 658/1000 [11:44<06:01,  1.06s/it]
Epoch Progress:  66%|███████████████▊        | 659/1000 [11:45<06:00,  1.06s/it]
Epoch Progress:  66%|███████████████▊        | 660/1000 [11:46<05:59,  1.06s/it]Dataset Size: 1%, Epoch: 661, Train Loss: 1.583162933588028, Val Loss: 12.12389280857184

Epoch Progress:  66%|███████████████▊        | 661/1000 [11:47<05:57,  1.06s/it]
Epoch Progress:  66%|███████████████▉        | 662/1000 [11:48<05:56,  1.06s/it]
Epoch Progress:  66%|███████████████▉        | 663/1000 [11:49<05:56,  1.06s/it]
Epoch Progress:  66%|███████████████▉        | 664/1000 [11:50<05:55,  1.06s/it]
Epoch Progress:  66%|███████████████▉        | 665/1000 [11:52<05:54,  1.06s/it]
Epoch Progress:  67%|███████████████▉        | 666/1000 [11:53<05:52,  1.06s/it]
Epoch Progress:  67%|████████████████        | 667/1000 [11:54<05:51,  1.06s/it]
Epoch Progress:  67%|████████████████        | 668/1000 [11:55<05:50,  1.06s/it]
Epoch Progress:  67%|████████████████        | 669/1000 [11:56<05:49,  1.06s/it]
Epoch Progress:  67%|████████████████        | 670/1000 [11:57<05:48,  1.06s/it]Dataset Size: 1%, Epoch: 671, Train Loss: 1.5838336944580078, Val Loss: 12.152485871926332

Epoch Progress:  67%|████████████████        | 671/1000 [11:58<05:46,  1.05s/it]
Epoch Progress:  67%|████████████████▏       | 672/1000 [11:59<05:46,  1.06s/it]
Epoch Progress:  67%|████████████████▏       | 673/1000 [12:00<05:46,  1.06s/it]
Epoch Progress:  67%|████████████████▏       | 674/1000 [12:01<05:45,  1.06s/it]
Epoch Progress:  68%|████████████████▏       | 675/1000 [12:02<05:43,  1.06s/it]
Epoch Progress:  68%|████████████████▏       | 676/1000 [12:03<05:42,  1.06s/it]
Epoch Progress:  68%|████████████████▏       | 677/1000 [12:04<05:40,  1.06s/it]
Epoch Progress:  68%|████████████████▎       | 678/1000 [12:05<05:40,  1.06s/it]
Epoch Progress:  68%|████████████████▎       | 679/1000 [12:06<05:39,  1.06s/it]
Epoch Progress:  68%|████████████████▎       | 680/1000 [12:07<05:38,  1.06s/it]Dataset Size: 1%, Epoch: 681, Train Loss: 1.5764228403568268, Val Loss: 12.183744577261118

Epoch Progress:  68%|████████████████▎       | 681/1000 [12:08<05:37,  1.06s/it]
Epoch Progress:  68%|████████████████▎       | 682/1000 [12:09<05:36,  1.06s/it]
Epoch Progress:  68%|████████████████▍       | 683/1000 [12:11<05:35,  1.06s/it]
Epoch Progress:  68%|████████████████▍       | 684/1000 [12:12<05:33,  1.06s/it]
Epoch Progress:  68%|████████████████▍       | 685/1000 [12:13<05:32,  1.06s/it]
Epoch Progress:  69%|████████████████▍       | 686/1000 [12:14<05:32,  1.06s/it]
Epoch Progress:  69%|████████████████▍       | 687/1000 [12:15<05:31,  1.06s/it]
Epoch Progress:  69%|████████████████▌       | 688/1000 [12:16<05:30,  1.06s/it]
Epoch Progress:  69%|████████████████▌       | 689/1000 [12:17<05:28,  1.06s/it]
Epoch Progress:  69%|████████████████▌       | 690/1000 [12:18<05:27,  1.06s/it]Dataset Size: 1%, Epoch: 691, Train Loss: 1.5721343159675598, Val Loss: 12.21271108969664

Epoch Progress:  69%|████████████████▌       | 691/1000 [12:19<05:26,  1.06s/it]
Epoch Progress:  69%|████████████████▌       | 692/1000 [12:20<05:25,  1.06s/it]
Epoch Progress:  69%|████████████████▋       | 693/1000 [12:21<05:24,  1.06s/it]
Epoch Progress:  69%|████████████████▋       | 694/1000 [12:22<05:23,  1.06s/it]
Epoch Progress:  70%|████████████████▋       | 695/1000 [12:23<05:22,  1.06s/it]
Epoch Progress:  70%|████████████████▋       | 696/1000 [12:24<05:21,  1.06s/it]
Epoch Progress:  70%|████████████████▋       | 697/1000 [12:25<05:20,  1.06s/it]
Epoch Progress:  70%|████████████████▊       | 698/1000 [12:26<05:19,  1.06s/it]
Epoch Progress:  70%|████████████████▊       | 699/1000 [12:27<05:18,  1.06s/it]
Epoch Progress:  70%|████████████████▊       | 700/1000 [12:29<05:17,  1.06s/it]Dataset Size: 1%, Epoch: 701, Train Loss: 1.5639658868312836, Val Loss: 12.23831059382512

Epoch Progress:  70%|████████████████▊       | 701/1000 [12:30<05:17,  1.06s/it]
Epoch Progress:  70%|████████████████▊       | 702/1000 [12:31<05:16,  1.06s/it]
Epoch Progress:  70%|████████████████▊       | 703/1000 [12:32<05:14,  1.06s/it]
Epoch Progress:  70%|████████████████▉       | 704/1000 [12:33<05:13,  1.06s/it]
Epoch Progress:  70%|████████████████▉       | 705/1000 [12:34<05:11,  1.06s/it]
Epoch Progress:  71%|████████████████▉       | 706/1000 [12:35<05:10,  1.05s/it]
Epoch Progress:  71%|████████████████▉       | 707/1000 [12:36<05:08,  1.05s/it]
Epoch Progress:  71%|████████████████▉       | 708/1000 [12:37<05:07,  1.05s/it]
Epoch Progress:  71%|█████████████████       | 709/1000 [12:38<05:07,  1.06s/it]
Epoch Progress:  71%|█████████████████       | 710/1000 [12:39<05:05,  1.05s/it]Dataset Size: 1%, Epoch: 711, Train Loss: 1.5604095757007599, Val Loss: 12.256538073221842

Epoch Progress:  71%|█████████████████       | 711/1000 [12:40<05:05,  1.06s/it]
Epoch Progress:  71%|█████████████████       | 712/1000 [12:41<05:04,  1.06s/it]
Epoch Progress:  71%|█████████████████       | 713/1000 [12:42<05:03,  1.06s/it]
Epoch Progress:  71%|█████████████████▏      | 714/1000 [12:43<05:01,  1.05s/it]
Epoch Progress:  72%|█████████████████▏      | 715/1000 [12:44<05:00,  1.05s/it]
Epoch Progress:  72%|█████████████████▏      | 716/1000 [12:45<05:00,  1.06s/it]
Epoch Progress:  72%|█████████████████▏      | 717/1000 [12:46<04:59,  1.06s/it]
Epoch Progress:  72%|█████████████████▏      | 718/1000 [12:48<04:58,  1.06s/it]
Epoch Progress:  72%|█████████████████▎      | 719/1000 [12:49<04:56,  1.06s/it]
Epoch Progress:  72%|█████████████████▎      | 720/1000 [12:50<04:56,  1.06s/it]Dataset Size: 1%, Epoch: 721, Train Loss: 1.550025075674057, Val Loss: 12.282086054484049

Epoch Progress:  72%|█████████████████▎      | 721/1000 [12:51<04:54,  1.06s/it]
Epoch Progress:  72%|█████████████████▎      | 722/1000 [12:52<04:53,  1.05s/it]
Epoch Progress:  72%|█████████████████▎      | 723/1000 [12:53<04:51,  1.05s/it]
Epoch Progress:  72%|█████████████████▍      | 724/1000 [12:54<04:50,  1.05s/it]
Epoch Progress:  72%|█████████████████▍      | 725/1000 [12:55<04:49,  1.05s/it]
Epoch Progress:  73%|█████████████████▍      | 726/1000 [12:56<04:48,  1.05s/it]
Epoch Progress:  73%|█████████████████▍      | 727/1000 [12:57<04:47,  1.05s/it]
Epoch Progress:  73%|█████████████████▍      | 728/1000 [12:58<04:46,  1.05s/it]
Epoch Progress:  73%|█████████████████▍      | 729/1000 [12:59<04:45,  1.05s/it]
Epoch Progress:  73%|█████████████████▌      | 730/1000 [13:00<04:46,  1.06s/it]Dataset Size: 1%, Epoch: 731, Train Loss: 1.5494594275951385, Val Loss: 12.306029686560997

Epoch Progress:  73%|█████████████████▌      | 731/1000 [13:01<04:45,  1.06s/it]
Epoch Progress:  73%|█████████████████▌      | 732/1000 [13:02<04:44,  1.06s/it]
Epoch Progress:  73%|█████████████████▌      | 733/1000 [13:03<04:42,  1.06s/it]
Epoch Progress:  73%|█████████████████▌      | 734/1000 [13:04<04:41,  1.06s/it]
Epoch Progress:  74%|█████████████████▋      | 735/1000 [13:05<04:40,  1.06s/it]
Epoch Progress:  74%|█████████████████▋      | 736/1000 [13:07<04:39,  1.06s/it]
Epoch Progress:  74%|█████████████████▋      | 737/1000 [13:08<04:38,  1.06s/it]
Epoch Progress:  74%|█████████████████▋      | 738/1000 [13:09<04:36,  1.06s/it]
Epoch Progress:  74%|█████████████████▋      | 739/1000 [13:10<04:36,  1.06s/it]
Epoch Progress:  74%|█████████████████▊      | 740/1000 [13:11<04:35,  1.06s/it]Dataset Size: 1%, Epoch: 741, Train Loss: 1.5412113964557648, Val Loss: 12.333477729406113

Epoch Progress:  74%|█████████████████▊      | 741/1000 [13:12<04:34,  1.06s/it]
Epoch Progress:  74%|█████████████████▊      | 742/1000 [13:13<04:32,  1.06s/it]
Epoch Progress:  74%|█████████████████▊      | 743/1000 [13:14<04:31,  1.06s/it]
Epoch Progress:  74%|█████████████████▊      | 744/1000 [13:15<04:30,  1.06s/it]
Epoch Progress:  74%|█████████████████▉      | 745/1000 [13:16<04:29,  1.06s/it]
Epoch Progress:  75%|█████████████████▉      | 746/1000 [13:17<04:29,  1.06s/it]
Epoch Progress:  75%|█████████████████▉      | 747/1000 [13:18<04:28,  1.06s/it]
Epoch Progress:  75%|█████████████████▉      | 748/1000 [13:19<04:26,  1.06s/it]
Epoch Progress:  75%|█████████████████▉      | 749/1000 [13:20<04:25,  1.06s/it]
Epoch Progress:  75%|██████████████████      | 750/1000 [13:21<04:24,  1.06s/it]Dataset Size: 1%, Epoch: 751, Train Loss: 1.537558138370514, Val Loss: 12.352695513994266

Epoch Progress:  75%|██████████████████      | 751/1000 [13:22<04:22,  1.06s/it]
Epoch Progress:  75%|██████████████████      | 752/1000 [13:23<04:21,  1.06s/it]
Epoch Progress:  75%|██████████████████      | 753/1000 [13:25<04:20,  1.06s/it]
Epoch Progress:  75%|██████████████████      | 754/1000 [13:26<04:19,  1.05s/it]
Epoch Progress:  76%|██████████████████      | 755/1000 [13:27<04:18,  1.05s/it]
Epoch Progress:  76%|██████████████████▏     | 756/1000 [13:28<04:17,  1.06s/it]
Epoch Progress:  76%|██████████████████▏     | 757/1000 [13:29<04:16,  1.06s/it]
Epoch Progress:  76%|██████████████████▏     | 758/1000 [13:30<04:17,  1.06s/it]
Epoch Progress:  76%|██████████████████▏     | 759/1000 [13:31<04:15,  1.06s/it]
Epoch Progress:  76%|██████████████████▏     | 760/1000 [13:32<04:14,  1.06s/it]Dataset Size: 1%, Epoch: 761, Train Loss: 1.5378054976463318, Val Loss: 12.367287391271347

Epoch Progress:  76%|██████████████████▎     | 761/1000 [13:33<04:12,  1.06s/it]
Epoch Progress:  76%|██████████████████▎     | 762/1000 [13:34<04:11,  1.06s/it]
Epoch Progress:  76%|██████████████████▎     | 763/1000 [13:35<04:10,  1.06s/it]
Epoch Progress:  76%|██████████████████▎     | 764/1000 [13:36<04:09,  1.06s/it]
Epoch Progress:  76%|██████████████████▎     | 765/1000 [13:37<04:07,  1.06s/it]
Epoch Progress:  77%|██████████████████▍     | 766/1000 [13:38<04:06,  1.05s/it]
Epoch Progress:  77%|██████████████████▍     | 767/1000 [13:39<04:05,  1.05s/it]
Epoch Progress:  77%|██████████████████▍     | 768/1000 [13:40<04:05,  1.06s/it]
Epoch Progress:  77%|██████████████████▍     | 769/1000 [13:41<04:03,  1.06s/it]
Epoch Progress:  77%|██████████████████▍     | 770/1000 [13:42<04:02,  1.05s/it]Dataset Size: 1%, Epoch: 771, Train Loss: 1.5346968173980713, Val Loss: 12.380733905694424

Epoch Progress:  77%|██████████████████▌     | 771/1000 [13:44<04:01,  1.05s/it]
Epoch Progress:  77%|██████████████████▌     | 772/1000 [13:45<04:00,  1.05s/it]
Epoch Progress:  77%|██████████████████▌     | 773/1000 [13:46<03:59,  1.05s/it]
Epoch Progress:  77%|██████████████████▌     | 774/1000 [13:47<03:58,  1.05s/it]
Epoch Progress:  78%|██████████████████▌     | 775/1000 [13:48<03:56,  1.05s/it]
Epoch Progress:  78%|██████████████████▌     | 776/1000 [13:49<03:56,  1.06s/it]
Epoch Progress:  78%|██████████████████▋     | 777/1000 [13:50<03:57,  1.06s/it]
Epoch Progress:  78%|██████████████████▋     | 778/1000 [13:51<03:55,  1.06s/it]
Epoch Progress:  78%|██████████████████▋     | 779/1000 [13:52<03:53,  1.06s/it]
Epoch Progress:  78%|██████████████████▋     | 780/1000 [13:53<03:52,  1.06s/it]Dataset Size: 1%, Epoch: 781, Train Loss: 1.5284216403961182, Val Loss: 12.397736378205128

Epoch Progress:  78%|██████████████████▋     | 781/1000 [13:54<03:51,  1.06s/it]
Epoch Progress:  78%|██████████████████▊     | 782/1000 [13:55<03:49,  1.05s/it]
Epoch Progress:  78%|██████████████████▊     | 783/1000 [13:56<03:48,  1.05s/it]
Epoch Progress:  78%|██████████████████▊     | 784/1000 [13:57<03:47,  1.05s/it]
Epoch Progress:  78%|██████████████████▊     | 785/1000 [13:58<03:46,  1.05s/it]
Epoch Progress:  79%|██████████████████▊     | 786/1000 [13:59<03:45,  1.06s/it]
Epoch Progress:  79%|██████████████████▉     | 787/1000 [14:00<03:44,  1.06s/it]
Epoch Progress:  79%|██████████████████▉     | 788/1000 [14:01<03:43,  1.05s/it]
Epoch Progress:  79%|██████████████████▉     | 789/1000 [14:03<03:42,  1.05s/it]
Epoch Progress:  79%|██████████████████▉     | 790/1000 [14:04<03:41,  1.05s/it]Dataset Size: 1%, Epoch: 791, Train Loss: 1.520700842142105, Val Loss: 12.413610971890963

Epoch Progress:  79%|██████████████████▉     | 791/1000 [14:05<03:40,  1.05s/it]
Epoch Progress:  79%|███████████████████     | 792/1000 [14:06<03:39,  1.06s/it]
Epoch Progress:  79%|███████████████████     | 793/1000 [14:07<03:38,  1.05s/it]
Epoch Progress:  79%|███████████████████     | 794/1000 [14:08<03:37,  1.05s/it]
Epoch Progress:  80%|███████████████████     | 795/1000 [14:09<03:36,  1.05s/it]
Epoch Progress:  80%|███████████████████     | 796/1000 [14:10<03:35,  1.06s/it]
Epoch Progress:  80%|███████████████████▏    | 797/1000 [14:11<03:34,  1.06s/it]
Epoch Progress:  80%|███████████████████▏    | 798/1000 [14:12<03:33,  1.06s/it]
Epoch Progress:  80%|███████████████████▏    | 799/1000 [14:13<03:31,  1.05s/it]
Epoch Progress:  80%|███████████████████▏    | 800/1000 [14:14<03:30,  1.05s/it]Dataset Size: 1%, Epoch: 801, Train Loss: 1.5269269049167633, Val Loss: 12.428589625236315

Epoch Progress:  80%|███████████████████▏    | 801/1000 [14:15<03:29,  1.05s/it]
Epoch Progress:  80%|███████████████████▏    | 802/1000 [14:16<03:28,  1.05s/it]
Epoch Progress:  80%|███████████████████▎    | 803/1000 [14:17<03:27,  1.05s/it]
Epoch Progress:  80%|███████████████████▎    | 804/1000 [14:18<03:26,  1.05s/it]
Epoch Progress:  80%|███████████████████▎    | 805/1000 [14:19<03:25,  1.05s/it]
Epoch Progress:  81%|███████████████████▎    | 806/1000 [14:20<03:25,  1.06s/it]
Epoch Progress:  81%|███████████████████▎    | 807/1000 [14:22<03:24,  1.06s/it]
Epoch Progress:  81%|███████████████████▍    | 808/1000 [14:23<03:23,  1.06s/it]
Epoch Progress:  81%|███████████████████▍    | 809/1000 [14:24<03:21,  1.06s/it]
Epoch Progress:  81%|███████████████████▍    | 810/1000 [14:25<03:20,  1.06s/it]Dataset Size: 1%, Epoch: 811, Train Loss: 1.5158774554729462, Val Loss: 12.440202492934008

Epoch Progress:  81%|███████████████████▍    | 811/1000 [14:26<03:19,  1.05s/it]
Epoch Progress:  81%|███████████████████▍    | 812/1000 [14:27<03:18,  1.05s/it]
Epoch Progress:  81%|███████████████████▌    | 813/1000 [14:28<03:16,  1.05s/it]
Epoch Progress:  81%|███████████████████▌    | 814/1000 [14:29<03:15,  1.05s/it]
Epoch Progress:  82%|███████████████████▌    | 815/1000 [14:30<03:15,  1.05s/it]
Epoch Progress:  82%|███████████████████▌    | 816/1000 [14:31<03:13,  1.05s/it]
Epoch Progress:  82%|███████████████████▌    | 817/1000 [14:32<03:12,  1.05s/it]
Epoch Progress:  82%|███████████████████▋    | 818/1000 [14:33<03:11,  1.05s/it]
Epoch Progress:  82%|███████████████████▋    | 819/1000 [14:34<03:10,  1.05s/it]
Epoch Progress:  82%|███████████████████▋    | 820/1000 [14:35<03:09,  1.05s/it]Dataset Size: 1%, Epoch: 821, Train Loss: 1.5191650092601776, Val Loss: 12.444810207073505

Epoch Progress:  82%|███████████████████▋    | 821/1000 [14:36<03:08,  1.05s/it]
Epoch Progress:  82%|███████████████████▋    | 822/1000 [14:37<03:07,  1.05s/it]
Epoch Progress:  82%|███████████████████▊    | 823/1000 [14:38<03:06,  1.05s/it]
Epoch Progress:  82%|███████████████████▊    | 824/1000 [14:39<03:05,  1.06s/it]
Epoch Progress:  82%|███████████████████▊    | 825/1000 [14:40<03:04,  1.06s/it]
Epoch Progress:  83%|███████████████████▊    | 826/1000 [14:42<03:03,  1.06s/it]
Epoch Progress:  83%|███████████████████▊    | 827/1000 [14:43<03:02,  1.05s/it]
Epoch Progress:  83%|███████████████████▊    | 828/1000 [14:44<03:01,  1.05s/it]
Epoch Progress:  83%|███████████████████▉    | 829/1000 [14:45<03:00,  1.05s/it]
Epoch Progress:  83%|███████████████████▉    | 830/1000 [14:46<02:59,  1.05s/it]Dataset Size: 1%, Epoch: 831, Train Loss: 1.5191838145256042, Val Loss: 12.447886907137358

Epoch Progress:  83%|███████████████████▉    | 831/1000 [14:47<02:58,  1.05s/it]
Epoch Progress:  83%|███████████████████▉    | 832/1000 [14:48<02:57,  1.05s/it]
Epoch Progress:  83%|███████████████████▉    | 833/1000 [14:49<02:56,  1.05s/it]
Epoch Progress:  83%|████████████████████    | 834/1000 [14:50<02:55,  1.06s/it]
Epoch Progress:  84%|████████████████████    | 835/1000 [14:51<02:54,  1.06s/it]
Epoch Progress:  84%|████████████████████    | 836/1000 [14:52<02:53,  1.06s/it]
Epoch Progress:  84%|████████████████████    | 837/1000 [14:53<02:52,  1.06s/it]
Epoch Progress:  84%|████████████████████    | 838/1000 [14:54<02:51,  1.06s/it]
Epoch Progress:  84%|████████████████████▏   | 839/1000 [14:55<02:50,  1.06s/it]
Epoch Progress:  84%|████████████████████▏   | 840/1000 [14:56<02:48,  1.06s/it]Dataset Size: 1%, Epoch: 841, Train Loss: 1.5158422589302063, Val Loss: 12.473794888227413

Epoch Progress:  84%|████████████████████▏   | 841/1000 [14:57<02:47,  1.06s/it]
Epoch Progress:  84%|████████████████████▏   | 842/1000 [14:58<02:46,  1.06s/it]
Epoch Progress:  84%|████████████████████▏   | 843/1000 [15:00<02:46,  1.06s/it]
Epoch Progress:  84%|████████████████████▎   | 844/1000 [15:01<02:45,  1.06s/it]
Epoch Progress:  84%|████████████████████▎   | 845/1000 [15:02<02:43,  1.06s/it]
Epoch Progress:  85%|████████████████████▎   | 846/1000 [15:03<02:42,  1.06s/it]
Epoch Progress:  85%|████████████████████▎   | 847/1000 [15:04<02:41,  1.06s/it]
Epoch Progress:  85%|████████████████████▎   | 848/1000 [15:05<02:40,  1.06s/it]
Epoch Progress:  85%|████████████████████▍   | 849/1000 [15:06<02:39,  1.06s/it]
Epoch Progress:  85%|████████████████████▍   | 850/1000 [15:07<02:38,  1.05s/it]Dataset Size: 1%, Epoch: 851, Train Loss: 1.5170833468437195, Val Loss: 12.47506046295166

Epoch Progress:  85%|████████████████████▍   | 851/1000 [15:08<02:37,  1.05s/it]
Epoch Progress:  85%|████████████████████▍   | 852/1000 [15:09<02:36,  1.05s/it]
Epoch Progress:  85%|████████████████████▍   | 853/1000 [15:10<02:35,  1.06s/it]
Epoch Progress:  85%|████████████████████▍   | 854/1000 [15:11<02:34,  1.06s/it]
Epoch Progress:  86%|████████████████████▌   | 855/1000 [15:12<02:33,  1.06s/it]
Epoch Progress:  86%|████████████████████▌   | 856/1000 [15:13<02:33,  1.06s/it]
Epoch Progress:  86%|████████████████████▌   | 857/1000 [15:14<02:32,  1.06s/it]
Epoch Progress:  86%|████████████████████▌   | 858/1000 [15:15<02:30,  1.06s/it]
Epoch Progress:  86%|████████████████████▌   | 859/1000 [15:16<02:29,  1.06s/it]
Epoch Progress:  86%|████████████████████▋   | 860/1000 [15:17<02:27,  1.06s/it]Dataset Size: 1%, Epoch: 861, Train Loss: 1.5088593065738678, Val Loss: 12.48975159571721

Epoch Progress:  86%|████████████████████▋   | 861/1000 [15:19<02:26,  1.05s/it]
Epoch Progress:  86%|████████████████████▋   | 862/1000 [15:20<02:26,  1.06s/it]
Epoch Progress:  86%|████████████████████▋   | 863/1000 [15:21<02:24,  1.06s/it]
Epoch Progress:  86%|████████████████████▋   | 864/1000 [15:22<02:23,  1.06s/it]
Epoch Progress:  86%|████████████████████▊   | 865/1000 [15:23<02:22,  1.06s/it]
Epoch Progress:  87%|████████████████████▊   | 866/1000 [15:24<02:22,  1.06s/it]
Epoch Progress:  87%|████████████████████▊   | 867/1000 [15:25<02:21,  1.06s/it]
Epoch Progress:  87%|████████████████████▊   | 868/1000 [15:26<02:20,  1.06s/it]
Epoch Progress:  87%|████████████████████▊   | 869/1000 [15:27<02:18,  1.06s/it]
Epoch Progress:  87%|████████████████████▉   | 870/1000 [15:28<02:17,  1.06s/it]Dataset Size: 1%, Epoch: 871, Train Loss: 1.505058377981186, Val Loss: 12.494061616750864

Epoch Progress:  87%|████████████████████▉   | 871/1000 [15:29<02:16,  1.06s/it]
Epoch Progress:  87%|████████████████████▉   | 872/1000 [15:30<02:15,  1.06s/it]
Epoch Progress:  87%|████████████████████▉   | 873/1000 [15:31<02:14,  1.06s/it]
Epoch Progress:  87%|████████████████████▉   | 874/1000 [15:32<02:12,  1.05s/it]
Epoch Progress:  88%|█████████████████████   | 875/1000 [15:33<02:11,  1.05s/it]
Epoch Progress:  88%|█████████████████████   | 876/1000 [15:34<02:10,  1.05s/it]
Epoch Progress:  88%|█████████████████████   | 877/1000 [15:35<02:09,  1.05s/it]
Epoch Progress:  88%|█████████████████████   | 878/1000 [15:37<02:08,  1.05s/it]
Epoch Progress:  88%|█████████████████████   | 879/1000 [15:38<02:07,  1.05s/it]
Epoch Progress:  88%|█████████████████████   | 880/1000 [15:39<02:06,  1.05s/it]Dataset Size: 1%, Epoch: 881, Train Loss: 1.5060452222824097, Val Loss: 12.499216104165102

Epoch Progress:  88%|█████████████████████▏  | 881/1000 [15:40<02:05,  1.06s/it]
Epoch Progress:  88%|█████████████████████▏  | 882/1000 [15:41<02:04,  1.06s/it]
Epoch Progress:  88%|█████████████████████▏  | 883/1000 [15:42<02:03,  1.05s/it]
Epoch Progress:  88%|█████████████████████▏  | 884/1000 [15:43<02:02,  1.05s/it]
Epoch Progress:  88%|█████████████████████▏  | 885/1000 [15:44<02:01,  1.05s/it]
Epoch Progress:  89%|█████████████████████▎  | 886/1000 [15:45<02:00,  1.05s/it]
Epoch Progress:  89%|█████████████████████▎  | 887/1000 [15:46<01:59,  1.05s/it]
Epoch Progress:  89%|█████████████████████▎  | 888/1000 [15:47<01:57,  1.05s/it]
Epoch Progress:  89%|█████████████████████▎  | 889/1000 [15:48<01:57,  1.05s/it]
Epoch Progress:  89%|█████████████████████▎  | 890/1000 [15:49<01:56,  1.06s/it]Dataset Size: 1%, Epoch: 891, Train Loss: 1.5058620274066925, Val Loss: 12.50598298586332

Epoch Progress:  89%|█████████████████████▍  | 891/1000 [15:50<01:55,  1.06s/it]
Epoch Progress:  89%|█████████████████████▍  | 892/1000 [15:51<01:54,  1.06s/it]
Epoch Progress:  89%|█████████████████████▍  | 893/1000 [15:52<01:53,  1.06s/it]
Epoch Progress:  89%|█████████████████████▍  | 894/1000 [15:53<01:52,  1.06s/it]
Epoch Progress:  90%|█████████████████████▍  | 895/1000 [15:54<01:50,  1.05s/it]
Epoch Progress:  90%|█████████████████████▌  | 896/1000 [15:56<01:50,  1.06s/it]
Epoch Progress:  90%|█████████████████████▌  | 897/1000 [15:57<01:49,  1.06s/it]
Epoch Progress:  90%|█████████████████████▌  | 898/1000 [15:58<01:48,  1.06s/it]
Epoch Progress:  90%|█████████████████████▌  | 899/1000 [15:59<01:47,  1.06s/it]
Epoch Progress:  90%|█████████████████████▌  | 900/1000 [16:00<01:46,  1.06s/it]Dataset Size: 1%, Epoch: 901, Train Loss: 1.5096376538276672, Val Loss: 12.512871644435785

Epoch Progress:  90%|█████████████████████▌  | 901/1000 [16:01<01:45,  1.06s/it]
Epoch Progress:  90%|█████████████████████▋  | 902/1000 [16:02<01:43,  1.06s/it]
Epoch Progress:  90%|█████████████████████▋  | 903/1000 [16:03<01:42,  1.06s/it]
Epoch Progress:  90%|█████████████████████▋  | 904/1000 [16:04<01:41,  1.06s/it]
Epoch Progress:  90%|█████████████████████▋  | 905/1000 [16:05<01:40,  1.06s/it]
Epoch Progress:  91%|█████████████████████▋  | 906/1000 [16:06<01:39,  1.06s/it]
Epoch Progress:  91%|█████████████████████▊  | 907/1000 [16:07<01:38,  1.06s/it]
Epoch Progress:  91%|█████████████████████▊  | 908/1000 [16:08<01:37,  1.06s/it]
Epoch Progress:  91%|█████████████████████▊  | 909/1000 [16:09<01:35,  1.05s/it]
Epoch Progress:  91%|█████████████████████▊  | 910/1000 [16:10<01:35,  1.06s/it]Dataset Size: 1%, Epoch: 911, Train Loss: 1.5074958801269531, Val Loss: 12.513591203934107

Epoch Progress:  91%|█████████████████████▊  | 911/1000 [16:11<01:33,  1.06s/it]
Epoch Progress:  91%|█████████████████████▉  | 912/1000 [16:12<01:32,  1.06s/it]
Epoch Progress:  91%|█████████████████████▉  | 913/1000 [16:14<01:31,  1.05s/it]
Epoch Progress:  91%|█████████████████████▉  | 914/1000 [16:15<01:30,  1.05s/it]
Epoch Progress:  92%|█████████████████████▉  | 915/1000 [16:16<01:29,  1.05s/it]
Epoch Progress:  92%|█████████████████████▉  | 916/1000 [16:17<01:28,  1.05s/it]
Epoch Progress:  92%|██████████████████████  | 917/1000 [16:18<01:27,  1.05s/it]
Epoch Progress:  92%|██████████████████████  | 918/1000 [16:19<01:26,  1.05s/it]
Epoch Progress:  92%|██████████████████████  | 919/1000 [16:20<01:25,  1.05s/it]
Epoch Progress:  92%|██████████████████████  | 920/1000 [16:21<01:24,  1.05s/it]Dataset Size: 1%, Epoch: 921, Train Loss: 1.5109563767910004, Val Loss: 12.511222741542719

Epoch Progress:  92%|██████████████████████  | 921/1000 [16:22<01:23,  1.05s/it]
Epoch Progress:  92%|██████████████████████▏ | 922/1000 [16:23<01:22,  1.05s/it]
Epoch Progress:  92%|██████████████████████▏ | 923/1000 [16:24<01:21,  1.05s/it]
Epoch Progress:  92%|██████████████████████▏ | 924/1000 [16:25<01:20,  1.05s/it]
Epoch Progress:  92%|██████████████████████▏ | 925/1000 [16:26<01:18,  1.05s/it]
Epoch Progress:  93%|██████████████████████▏ | 926/1000 [16:27<01:18,  1.05s/it]
Epoch Progress:  93%|██████████████████████▏ | 927/1000 [16:28<01:17,  1.06s/it]
Epoch Progress:  93%|██████████████████████▎ | 928/1000 [16:29<01:16,  1.06s/it]
Epoch Progress:  93%|██████████████████████▎ | 929/1000 [16:30<01:15,  1.06s/it]
Epoch Progress:  93%|██████████████████████▎ | 930/1000 [16:31<01:13,  1.06s/it]Dataset Size: 1%, Epoch: 931, Train Loss: 1.5027157664299011, Val Loss: 12.51300486540183

Epoch Progress:  93%|██████████████████████▎ | 931/1000 [16:32<01:12,  1.05s/it]
Epoch Progress:  93%|██████████████████████▎ | 932/1000 [16:34<01:11,  1.05s/it]
Epoch Progress:  93%|██████████████████████▍ | 933/1000 [16:35<01:10,  1.05s/it]
Epoch Progress:  93%|██████████████████████▍ | 934/1000 [16:36<01:09,  1.05s/it]
Epoch Progress:  94%|██████████████████████▍ | 935/1000 [16:37<01:08,  1.05s/it]
Epoch Progress:  94%|██████████████████████▍ | 936/1000 [16:38<01:07,  1.05s/it]
Epoch Progress:  94%|██████████████████████▍ | 937/1000 [16:39<01:06,  1.05s/it]
Epoch Progress:  94%|██████████████████████▌ | 938/1000 [16:40<01:05,  1.06s/it]
Epoch Progress:  94%|██████████████████████▌ | 939/1000 [16:41<01:04,  1.06s/it]
Epoch Progress:  94%|██████████████████████▌ | 940/1000 [16:42<01:03,  1.05s/it]Dataset Size: 1%, Epoch: 941, Train Loss: 1.5059353411197662, Val Loss: 12.512520129864033

Epoch Progress:  94%|██████████████████████▌ | 941/1000 [16:43<01:02,  1.05s/it]
Epoch Progress:  94%|██████████████████████▌ | 942/1000 [16:44<01:01,  1.05s/it]
Epoch Progress:  94%|██████████████████████▋ | 943/1000 [16:45<00:59,  1.05s/it]
Epoch Progress:  94%|██████████████████████▋ | 944/1000 [16:46<00:58,  1.05s/it]
Epoch Progress:  94%|██████████████████████▋ | 945/1000 [16:47<00:57,  1.05s/it]
Epoch Progress:  95%|██████████████████████▋ | 946/1000 [16:48<00:56,  1.05s/it]
Epoch Progress:  95%|██████████████████████▋ | 947/1000 [16:49<00:55,  1.05s/it]
Epoch Progress:  95%|██████████████████████▊ | 948/1000 [16:50<00:54,  1.05s/it]
Epoch Progress:  95%|██████████████████████▊ | 949/1000 [16:51<00:53,  1.05s/it]
Epoch Progress:  95%|██████████████████████▊ | 950/1000 [16:52<00:52,  1.05s/it]Dataset Size: 1%, Epoch: 951, Train Loss: 1.5038481056690216, Val Loss: 12.51749447064522

Epoch Progress:  95%|██████████████████████▊ | 951/1000 [16:54<00:51,  1.05s/it]
Epoch Progress:  95%|██████████████████████▊ | 952/1000 [16:55<00:50,  1.05s/it]
Epoch Progress:  95%|██████████████████████▊ | 953/1000 [16:56<00:49,  1.05s/it]
Epoch Progress:  95%|██████████████████████▉ | 954/1000 [16:57<00:48,  1.05s/it]
Epoch Progress:  96%|██████████████████████▉ | 955/1000 [16:58<00:47,  1.05s/it]
Epoch Progress:  96%|██████████████████████▉ | 956/1000 [16:59<00:46,  1.05s/it]
Epoch Progress:  96%|██████████████████████▉ | 957/1000 [17:00<00:45,  1.06s/it]
Epoch Progress:  96%|██████████████████████▉ | 958/1000 [17:01<00:44,  1.06s/it]
Epoch Progress:  96%|███████████████████████ | 959/1000 [17:02<00:43,  1.06s/it]
Epoch Progress:  96%|███████████████████████ | 960/1000 [17:03<00:42,  1.06s/it]Dataset Size: 1%, Epoch: 961, Train Loss: 1.5038491487503052, Val Loss: 12.525315822699131

Epoch Progress:  96%|███████████████████████ | 961/1000 [17:04<00:41,  1.05s/it]
Epoch Progress:  96%|███████████████████████ | 962/1000 [17:05<00:40,  1.05s/it]
Epoch Progress:  96%|███████████████████████ | 963/1000 [17:06<00:39,  1.05s/it]
Epoch Progress:  96%|███████████████████████▏| 964/1000 [17:07<00:37,  1.05s/it]
Epoch Progress:  96%|███████████████████████▏| 965/1000 [17:08<00:36,  1.05s/it]
Epoch Progress:  97%|███████████████████████▏| 966/1000 [17:09<00:35,  1.05s/it]
Epoch Progress:  97%|███████████████████████▏| 967/1000 [17:10<00:34,  1.06s/it]
Epoch Progress:  97%|███████████████████████▏| 968/1000 [17:11<00:33,  1.05s/it]
Epoch Progress:  97%|███████████████████████▎| 969/1000 [17:13<00:32,  1.05s/it]
Epoch Progress:  97%|███████████████████████▎| 970/1000 [17:14<00:31,  1.05s/it]Dataset Size: 1%, Epoch: 971, Train Loss: 1.5038071870803833, Val Loss: 12.526920147431202

Epoch Progress:  97%|███████████████████████▎| 971/1000 [17:15<00:30,  1.05s/it]
Epoch Progress:  97%|███████████████████████▎| 972/1000 [17:16<00:29,  1.05s/it]
Epoch Progress:  97%|███████████████████████▎| 973/1000 [17:17<00:28,  1.05s/it]
Epoch Progress:  97%|███████████████████████▍| 974/1000 [17:18<00:27,  1.05s/it]
Epoch Progress:  98%|███████████████████████▍| 975/1000 [17:19<00:26,  1.05s/it]
Epoch Progress:  98%|███████████████████████▍| 976/1000 [17:20<00:25,  1.06s/it]
Epoch Progress:  98%|███████████████████████▍| 977/1000 [17:21<00:24,  1.06s/it]
Epoch Progress:  98%|███████████████████████▍| 978/1000 [17:22<00:23,  1.05s/it]
Epoch Progress:  98%|███████████████████████▍| 979/1000 [17:23<00:22,  1.05s/it]
Epoch Progress:  98%|███████████████████████▌| 980/1000 [17:24<00:21,  1.05s/it]Dataset Size: 1%, Epoch: 981, Train Loss: 1.4993477165699005, Val Loss: 12.518543977003832

Epoch Progress:  98%|███████████████████████▌| 981/1000 [17:25<00:19,  1.05s/it]
Epoch Progress:  98%|███████████████████████▌| 982/1000 [17:26<00:18,  1.05s/it]
Epoch Progress:  98%|███████████████████████▌| 983/1000 [17:27<00:17,  1.05s/it]
Epoch Progress:  98%|███████████████████████▌| 984/1000 [17:28<00:16,  1.05s/it]
Epoch Progress:  98%|███████████████████████▋| 985/1000 [17:29<00:15,  1.05s/it]
Epoch Progress:  99%|███████████████████████▋| 986/1000 [17:30<00:14,  1.05s/it]
Epoch Progress:  99%|███████████████████████▋| 987/1000 [17:32<00:13,  1.06s/it]
Epoch Progress:  99%|███████████████████████▋| 988/1000 [17:33<00:12,  1.06s/it]
Epoch Progress:  99%|███████████████████████▋| 989/1000 [17:34<00:11,  1.06s/it]
Epoch Progress:  99%|███████████████████████▊| 990/1000 [17:35<00:10,  1.06s/it]Dataset Size: 1%, Epoch: 991, Train Loss: 1.503486543893814, Val Loss: 12.51955071473733

Epoch Progress:  99%|███████████████████████▊| 991/1000 [17:36<00:09,  1.06s/it]
Epoch Progress:  99%|███████████████████████▊| 992/1000 [17:37<00:08,  1.06s/it]
Epoch Progress:  99%|███████████████████████▊| 993/1000 [17:38<00:07,  1.06s/it]
Epoch Progress:  99%|███████████████████████▊| 994/1000 [17:39<00:06,  1.06s/it]
Epoch Progress: 100%|███████████████████████▉| 995/1000 [17:40<00:05,  1.06s/it]
Epoch Progress: 100%|███████████████████████▉| 996/1000 [17:41<00:04,  1.06s/it]
Epoch Progress: 100%|███████████████████████▉| 997/1000 [17:42<00:03,  1.06s/it]
Epoch Progress: 100%|███████████████████████▉| 998/1000 [17:43<00:02,  1.06s/it]
Epoch Progress: 100%|███████████████████████▉| 999/1000 [17:44<00:01,  1.06s/it]
Epoch Progress: 100%|███████████████████████| 1000/1000 [17:45<00:00,  1.07s/it]
Dataset Size: 1%, Val Perplexity: 3505.673583984375
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