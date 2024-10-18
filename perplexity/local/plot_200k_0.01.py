import re
import pandas as pd

# Simulated input string containing the epoch data.
log_data = """
Epoch Progress:   0%|                                  | 0/1000 [00:00<?, ?it/s]Dataset Size: 1%, Epoch: 1, Train Loss: 11.002289414405823, Val Loss: 11.004814581437545

Epoch Progress:   0%|                          | 1/1000 [00:01<27:08,  1.63s/it]
Epoch Progress:   0%|                          | 2/1000 [00:02<23:14,  1.40s/it]
Epoch Progress:   0%|                          | 3/1000 [00:04<21:59,  1.32s/it]
Epoch Progress:   0%|                          | 4/1000 [00:05<21:28,  1.29s/it]
Epoch Progress:   0%|▏                         | 5/1000 [00:06<21:23,  1.29s/it]
Epoch Progress:   1%|▏                         | 6/1000 [00:07<21:19,  1.29s/it]
Epoch Progress:   1%|▏                         | 7/1000 [00:09<21:03,  1.27s/it]
Epoch Progress:   1%|▏                         | 8/1000 [00:10<21:00,  1.27s/it]
Epoch Progress:   1%|▏                         | 9/1000 [00:11<20:45,  1.26s/it]
Epoch Progress:   1%|▎                        | 10/1000 [00:12<20:47,  1.26s/it]Dataset Size: 1%, Epoch: 11, Train Loss: 10.992569923400879, Val Loss: 10.99849709597501

Epoch Progress:   1%|▎                        | 11/1000 [00:14<20:44,  1.26s/it]
Epoch Progress:   1%|▎                        | 12/1000 [00:15<20:39,  1.25s/it]
Epoch Progress:   1%|▎                        | 13/1000 [00:16<20:30,  1.25s/it]
Epoch Progress:   1%|▎                        | 14/1000 [00:17<20:28,  1.25s/it]
Epoch Progress:   2%|▍                        | 15/1000 [00:19<20:26,  1.24s/it]
Epoch Progress:   2%|▍                        | 16/1000 [00:20<20:26,  1.25s/it]
Epoch Progress:   2%|▍                        | 17/1000 [00:21<20:19,  1.24s/it]
Epoch Progress:   2%|▍                        | 18/1000 [00:22<20:12,  1.23s/it]
Epoch Progress:   2%|▍                        | 19/1000 [00:24<20:13,  1.24s/it]
Epoch Progress:   2%|▌                        | 20/1000 [00:25<20:17,  1.24s/it]Dataset Size: 1%, Epoch: 21, Train Loss: 10.968865275382996, Val Loss: 10.983888304078734

Epoch Progress:   2%|▌                        | 21/1000 [00:26<20:20,  1.25s/it]
Epoch Progress:   2%|▌                        | 22/1000 [00:27<20:27,  1.25s/it]
Epoch Progress:   2%|▌                        | 23/1000 [00:29<20:32,  1.26s/it]
Epoch Progress:   2%|▌                        | 24/1000 [00:30<20:41,  1.27s/it]
Epoch Progress:   2%|▋                        | 25/1000 [00:31<20:30,  1.26s/it]
Epoch Progress:   3%|▋                        | 26/1000 [00:32<20:22,  1.25s/it]
Epoch Progress:   3%|▋                        | 27/1000 [00:34<20:12,  1.25s/it]
Epoch Progress:   3%|▋                        | 28/1000 [00:35<20:05,  1.24s/it]
Epoch Progress:   3%|▋                        | 29/1000 [00:36<20:02,  1.24s/it]
Epoch Progress:   3%|▊                        | 30/1000 [00:37<20:05,  1.24s/it]Dataset Size: 1%, Epoch: 31, Train Loss: 10.934592604637146, Val Loss: 10.961563964942833

Epoch Progress:   3%|▊                        | 31/1000 [00:39<19:58,  1.24s/it]
Epoch Progress:   3%|▊                        | 32/1000 [00:40<20:05,  1.25s/it]
Epoch Progress:   3%|▊                        | 33/1000 [00:41<20:04,  1.25s/it]
Epoch Progress:   3%|▊                        | 34/1000 [00:42<20:00,  1.24s/it]
Epoch Progress:   4%|▉                        | 35/1000 [00:44<20:03,  1.25s/it]
Epoch Progress:   4%|▉                        | 36/1000 [00:45<20:16,  1.26s/it]
Epoch Progress:   4%|▉                        | 37/1000 [00:46<20:16,  1.26s/it]
Epoch Progress:   4%|▉                        | 38/1000 [00:47<20:18,  1.27s/it]
Epoch Progress:   4%|▉                        | 39/1000 [00:49<20:19,  1.27s/it]
Epoch Progress:   4%|█                        | 40/1000 [00:50<20:22,  1.27s/it]Dataset Size: 1%, Epoch: 41, Train Loss: 10.892967700958252, Val Loss: 10.936583766689548

Epoch Progress:   4%|█                        | 41/1000 [00:51<20:11,  1.26s/it]
Epoch Progress:   4%|█                        | 42/1000 [00:52<20:05,  1.26s/it]
Epoch Progress:   4%|█                        | 43/1000 [00:54<19:57,  1.25s/it]
Epoch Progress:   4%|█                        | 44/1000 [00:55<19:51,  1.25s/it]
Epoch Progress:   4%|█▏                       | 45/1000 [00:56<19:44,  1.24s/it]
Epoch Progress:   5%|█▏                       | 46/1000 [00:57<19:44,  1.24s/it]
Epoch Progress:   5%|█▏                       | 47/1000 [00:59<19:38,  1.24s/it]
Epoch Progress:   5%|█▏                       | 48/1000 [01:00<19:41,  1.24s/it]
Epoch Progress:   5%|█▏                       | 49/1000 [01:01<19:35,  1.24s/it]
Epoch Progress:   5%|█▎                       | 50/1000 [01:02<19:33,  1.23s/it]Dataset Size: 1%, Epoch: 51, Train Loss: 10.850879073143005, Val Loss: 10.911066600254603

Epoch Progress:   5%|█▎                       | 51/1000 [01:04<19:33,  1.24s/it]
Epoch Progress:   5%|█▎                       | 52/1000 [01:05<19:32,  1.24s/it]
Epoch Progress:   5%|█▎                       | 53/1000 [01:06<19:34,  1.24s/it]
Epoch Progress:   5%|█▎                       | 54/1000 [01:07<19:34,  1.24s/it]
Epoch Progress:   6%|█▍                       | 55/1000 [01:09<19:32,  1.24s/it]
Epoch Progress:   6%|█▍                       | 56/1000 [01:10<19:32,  1.24s/it]
Epoch Progress:   6%|█▍                       | 57/1000 [01:11<19:38,  1.25s/it]
Epoch Progress:   6%|█▍                       | 58/1000 [01:12<19:31,  1.24s/it]
Epoch Progress:   6%|█▍                       | 59/1000 [01:14<19:31,  1.25s/it]
Epoch Progress:   6%|█▌                       | 60/1000 [01:15<19:28,  1.24s/it]Dataset Size: 1%, Epoch: 61, Train Loss: 10.80445420742035, Val Loss: 10.887397939508611

Epoch Progress:   6%|█▌                       | 61/1000 [01:16<19:36,  1.25s/it]
Epoch Progress:   6%|█▌                       | 62/1000 [01:17<19:29,  1.25s/it]
Epoch Progress:   6%|█▌                       | 63/1000 [01:19<19:25,  1.24s/it]
Epoch Progress:   6%|█▌                       | 64/1000 [01:20<19:30,  1.25s/it]
Epoch Progress:   6%|█▋                       | 65/1000 [01:21<19:31,  1.25s/it]
Epoch Progress:   7%|█▋                       | 66/1000 [01:22<19:34,  1.26s/it]
Epoch Progress:   7%|█▋                       | 67/1000 [01:24<19:31,  1.26s/it]
Epoch Progress:   7%|█▋                       | 68/1000 [01:25<19:27,  1.25s/it]
Epoch Progress:   7%|█▋                       | 69/1000 [01:26<19:21,  1.25s/it]
Epoch Progress:   7%|█▊                       | 70/1000 [01:27<19:19,  1.25s/it]Dataset Size: 1%, Epoch: 71, Train Loss: 10.757479667663574, Val Loss: 10.8673704444588

Epoch Progress:   7%|█▊                       | 71/1000 [01:29<19:16,  1.24s/it]
Epoch Progress:   7%|█▊                       | 72/1000 [01:30<19:10,  1.24s/it]
Epoch Progress:   7%|█▊                       | 73/1000 [01:31<19:06,  1.24s/it]
Epoch Progress:   7%|█▊                       | 74/1000 [01:32<19:01,  1.23s/it]
Epoch Progress:   8%|█▉                       | 75/1000 [01:33<18:59,  1.23s/it]
Epoch Progress:   8%|█▉                       | 76/1000 [01:35<18:59,  1.23s/it]
Epoch Progress:   8%|█▉                       | 77/1000 [01:36<19:01,  1.24s/it]
Epoch Progress:   8%|█▉                       | 78/1000 [01:37<19:04,  1.24s/it]
Epoch Progress:   8%|█▉                       | 79/1000 [01:38<19:00,  1.24s/it]
Epoch Progress:   8%|██                       | 80/1000 [01:40<19:02,  1.24s/it]Dataset Size: 1%, Epoch: 81, Train Loss: 10.708742260932922, Val Loss: 10.848452208878157

Epoch Progress:   8%|██                       | 81/1000 [01:41<19:07,  1.25s/it]
Epoch Progress:   8%|██                       | 82/1000 [01:42<19:08,  1.25s/it]
Epoch Progress:   8%|██                       | 83/1000 [01:43<19:01,  1.24s/it]
Epoch Progress:   8%|██                       | 84/1000 [01:45<18:59,  1.24s/it]
Epoch Progress:   8%|██▏                      | 85/1000 [01:46<18:54,  1.24s/it]
Epoch Progress:   9%|██▏                      | 86/1000 [01:47<18:52,  1.24s/it]
Epoch Progress:   9%|██▏                      | 87/1000 [01:48<18:53,  1.24s/it]
Epoch Progress:   9%|██▏                      | 88/1000 [01:50<18:59,  1.25s/it]
Epoch Progress:   9%|██▏                      | 89/1000 [01:51<18:54,  1.25s/it]
Epoch Progress:   9%|██▎                      | 90/1000 [01:52<18:52,  1.24s/it]Dataset Size: 1%, Epoch: 91, Train Loss: 10.652320981025696, Val Loss: 10.829763845963912

Epoch Progress:   9%|██▎                      | 91/1000 [01:53<18:48,  1.24s/it]
Epoch Progress:   9%|██▎                      | 92/1000 [01:55<18:44,  1.24s/it]
Epoch Progress:   9%|██▎                      | 93/1000 [01:56<18:38,  1.23s/it]
Epoch Progress:   9%|██▎                      | 94/1000 [01:57<18:33,  1.23s/it]
Epoch Progress:  10%|██▍                      | 95/1000 [01:58<18:30,  1.23s/it]
Epoch Progress:  10%|██▍                      | 96/1000 [01:59<18:30,  1.23s/it]
Epoch Progress:  10%|██▍                      | 97/1000 [02:01<18:28,  1.23s/it]
Epoch Progress:  10%|██▍                      | 98/1000 [02:02<18:27,  1.23s/it]
Epoch Progress:  10%|██▍                      | 99/1000 [02:03<18:27,  1.23s/it]
Epoch Progress:  10%|██▍                     | 100/1000 [02:04<18:23,  1.23s/it]Dataset Size: 1%, Epoch: 101, Train Loss: 10.567268013954163, Val Loss: 10.81029024991122

Epoch Progress:  10%|██▍                     | 101/1000 [02:06<18:24,  1.23s/it]
Epoch Progress:  10%|██▍                     | 102/1000 [02:07<18:26,  1.23s/it]
Epoch Progress:  10%|██▍                     | 103/1000 [02:08<18:27,  1.23s/it]
Epoch Progress:  10%|██▍                     | 104/1000 [02:09<18:23,  1.23s/it]
Epoch Progress:  10%|██▌                     | 105/1000 [02:11<18:23,  1.23s/it]
Epoch Progress:  11%|██▌                     | 106/1000 [02:12<18:20,  1.23s/it]
Epoch Progress:  11%|██▌                     | 107/1000 [02:13<18:18,  1.23s/it]
Epoch Progress:  11%|██▌                     | 108/1000 [02:14<18:17,  1.23s/it]
Epoch Progress:  11%|██▌                     | 109/1000 [02:15<18:18,  1.23s/it]
Epoch Progress:  11%|██▋                     | 110/1000 [02:17<18:16,  1.23s/it]Dataset Size: 1%, Epoch: 111, Train Loss: 10.466742873191833, Val Loss: 10.78968096398688

Epoch Progress:  11%|██▋                     | 111/1000 [02:18<18:11,  1.23s/it]
Epoch Progress:  11%|██▋                     | 112/1000 [02:19<18:13,  1.23s/it]
Epoch Progress:  11%|██▋                     | 113/1000 [02:20<18:19,  1.24s/it]
Epoch Progress:  11%|██▋                     | 114/1000 [02:22<18:12,  1.23s/it]
Epoch Progress:  12%|██▊                     | 115/1000 [02:23<18:10,  1.23s/it]
Epoch Progress:  12%|██▊                     | 116/1000 [02:24<18:11,  1.23s/it]
Epoch Progress:  12%|██▊                     | 117/1000 [02:25<18:10,  1.23s/it]
Epoch Progress:  12%|██▊                     | 118/1000 [02:27<18:07,  1.23s/it]
Epoch Progress:  12%|██▊                     | 119/1000 [02:28<18:02,  1.23s/it]
Epoch Progress:  12%|██▉                     | 120/1000 [02:29<18:00,  1.23s/it]Dataset Size: 1%, Epoch: 121, Train Loss: 10.358967304229736, Val Loss: 10.763654845101494

Epoch Progress:  12%|██▉                     | 121/1000 [02:30<18:01,  1.23s/it]
Epoch Progress:  12%|██▉                     | 122/1000 [02:31<17:57,  1.23s/it]
Epoch Progress:  12%|██▉                     | 123/1000 [02:33<17:53,  1.22s/it]
Epoch Progress:  12%|██▉                     | 124/1000 [02:34<17:52,  1.22s/it]
Epoch Progress:  12%|███                     | 125/1000 [02:35<17:49,  1.22s/it]
Epoch Progress:  13%|███                     | 126/1000 [02:36<17:44,  1.22s/it]
Epoch Progress:  13%|███                     | 127/1000 [02:38<17:45,  1.22s/it]
Epoch Progress:  13%|███                     | 128/1000 [02:39<17:39,  1.22s/it]
Epoch Progress:  13%|███                     | 129/1000 [02:40<17:40,  1.22s/it]
Epoch Progress:  13%|███                     | 130/1000 [02:41<17:35,  1.21s/it]Dataset Size: 1%, Epoch: 131, Train Loss: 10.218307971954346, Val Loss: 10.74005938195563

Epoch Progress:  13%|███▏                    | 131/1000 [02:42<17:33,  1.21s/it]
Epoch Progress:  13%|███▏                    | 132/1000 [02:44<17:32,  1.21s/it]
Epoch Progress:  13%|███▏                    | 133/1000 [02:45<17:31,  1.21s/it]
Epoch Progress:  13%|███▏                    | 134/1000 [02:46<17:28,  1.21s/it]
Epoch Progress:  14%|███▏                    | 135/1000 [02:47<17:28,  1.21s/it]
Epoch Progress:  14%|███▎                    | 136/1000 [02:48<17:26,  1.21s/it]
Epoch Progress:  14%|███▎                    | 137/1000 [02:50<17:28,  1.21s/it]
Epoch Progress:  14%|███▎                    | 138/1000 [02:51<17:31,  1.22s/it]
Epoch Progress:  14%|███▎                    | 139/1000 [02:52<17:27,  1.22s/it]
Epoch Progress:  14%|███▎                    | 140/1000 [02:53<17:24,  1.21s/it]Dataset Size: 1%, Epoch: 141, Train Loss: 10.043058395385742, Val Loss: 10.712692037805335

Epoch Progress:  14%|███▍                    | 141/1000 [02:55<17:23,  1.22s/it]
Epoch Progress:  14%|███▍                    | 142/1000 [02:56<17:23,  1.22s/it]
Epoch Progress:  14%|███▍                    | 143/1000 [02:57<17:21,  1.22s/it]
Epoch Progress:  14%|███▍                    | 144/1000 [02:58<17:24,  1.22s/it]
Epoch Progress:  14%|███▍                    | 145/1000 [02:59<17:24,  1.22s/it]
Epoch Progress:  15%|███▌                    | 146/1000 [03:01<17:22,  1.22s/it]
Epoch Progress:  15%|███▌                    | 147/1000 [03:02<17:17,  1.22s/it]
Epoch Progress:  15%|███▌                    | 148/1000 [03:03<17:16,  1.22s/it]
Epoch Progress:  15%|███▌                    | 149/1000 [03:04<17:16,  1.22s/it]
Epoch Progress:  15%|███▌                    | 150/1000 [03:06<17:15,  1.22s/it]Dataset Size: 1%, Epoch: 151, Train Loss: 9.889996528625488, Val Loss: 10.684925166043369

Epoch Progress:  15%|███▌                    | 151/1000 [03:07<17:13,  1.22s/it]
Epoch Progress:  15%|███▋                    | 152/1000 [03:08<17:12,  1.22s/it]
Epoch Progress:  15%|███▋                    | 153/1000 [03:09<17:11,  1.22s/it]
Epoch Progress:  15%|███▋                    | 154/1000 [03:10<17:11,  1.22s/it]
Epoch Progress:  16%|███▋                    | 155/1000 [03:12<17:07,  1.22s/it]
Epoch Progress:  16%|███▋                    | 156/1000 [03:13<17:06,  1.22s/it]
Epoch Progress:  16%|███▊                    | 157/1000 [03:14<17:05,  1.22s/it]
Epoch Progress:  16%|███▊                    | 158/1000 [03:15<17:03,  1.22s/it]
Epoch Progress:  16%|███▊                    | 159/1000 [03:16<17:01,  1.21s/it]
Epoch Progress:  16%|███▊                    | 160/1000 [03:18<16:59,  1.21s/it]Dataset Size: 1%, Epoch: 161, Train Loss: 9.734656810760498, Val Loss: 10.670869765343603

Epoch Progress:  16%|███▊                    | 161/1000 [03:19<16:58,  1.21s/it]
Epoch Progress:  16%|███▉                    | 162/1000 [03:20<17:02,  1.22s/it]
Epoch Progress:  16%|███▉                    | 163/1000 [03:21<16:56,  1.21s/it]
Epoch Progress:  16%|███▉                    | 164/1000 [03:23<17:00,  1.22s/it]
Epoch Progress:  16%|███▉                    | 165/1000 [03:24<17:01,  1.22s/it]
Epoch Progress:  17%|███▉                    | 166/1000 [03:25<16:59,  1.22s/it]
Epoch Progress:  17%|████                    | 167/1000 [03:26<16:55,  1.22s/it]
Epoch Progress:  17%|████                    | 168/1000 [03:27<16:53,  1.22s/it]
Epoch Progress:  17%|████                    | 169/1000 [03:29<16:48,  1.21s/it]
Epoch Progress:  17%|████                    | 170/1000 [03:30<16:53,  1.22s/it]Dataset Size: 1%, Epoch: 171, Train Loss: 9.560560941696167, Val Loss: 10.65906388419015

Epoch Progress:  17%|████                    | 171/1000 [03:31<16:49,  1.22s/it]
Epoch Progress:  17%|████▏                   | 172/1000 [03:32<16:45,  1.21s/it]
Epoch Progress:  17%|████▏                   | 173/1000 [03:33<16:42,  1.21s/it]
Epoch Progress:  17%|████▏                   | 174/1000 [03:35<16:39,  1.21s/it]
Epoch Progress:  18%|████▏                   | 175/1000 [03:36<16:40,  1.21s/it]
Epoch Progress:  18%|████▏                   | 176/1000 [03:37<16:37,  1.21s/it]
Epoch Progress:  18%|████▏                   | 177/1000 [03:38<16:36,  1.21s/it]
Epoch Progress:  18%|████▎                   | 178/1000 [03:40<16:38,  1.21s/it]
Epoch Progress:  18%|████▎                   | 179/1000 [03:41<16:35,  1.21s/it]
Epoch Progress:  18%|████▎                   | 180/1000 [03:42<16:31,  1.21s/it]Dataset Size: 1%, Epoch: 181, Train Loss: 9.41780936717987, Val Loss: 10.64000382361474

Epoch Progress:  18%|████▎                   | 181/1000 [03:43<16:30,  1.21s/it]
Epoch Progress:  18%|████▎                   | 182/1000 [03:44<16:30,  1.21s/it]
Epoch Progress:  18%|████▍                   | 183/1000 [03:46<16:30,  1.21s/it]
Epoch Progress:  18%|████▍                   | 184/1000 [03:47<16:30,  1.21s/it]
Epoch Progress:  18%|████▍                   | 185/1000 [03:48<16:28,  1.21s/it]
Epoch Progress:  19%|████▍                   | 186/1000 [03:49<16:26,  1.21s/it]
Epoch Progress:  19%|████▍                   | 187/1000 [03:50<16:25,  1.21s/it]
Epoch Progress:  19%|████▌                   | 188/1000 [03:52<16:23,  1.21s/it]
Epoch Progress:  19%|████▌                   | 189/1000 [03:53<16:21,  1.21s/it]
Epoch Progress:  19%|████▌                   | 190/1000 [03:54<16:36,  1.23s/it]Dataset Size: 1%, Epoch: 191, Train Loss: 9.258114099502563, Val Loss: 10.649325965286849

Epoch Progress:  19%|████▌                   | 191/1000 [03:55<16:34,  1.23s/it]
Epoch Progress:  19%|████▌                   | 192/1000 [03:57<16:26,  1.22s/it]
Epoch Progress:  19%|████▋                   | 193/1000 [03:58<16:22,  1.22s/it]
Epoch Progress:  19%|████▋                   | 194/1000 [03:59<16:17,  1.21s/it]
Epoch Progress:  20%|████▋                   | 195/1000 [04:00<16:15,  1.21s/it]
Epoch Progress:  20%|████▋                   | 196/1000 [04:01<16:13,  1.21s/it]
Epoch Progress:  20%|████▋                   | 197/1000 [04:03<16:12,  1.21s/it]
Epoch Progress:  20%|████▊                   | 198/1000 [04:04<16:11,  1.21s/it]
Epoch Progress:  20%|████▊                   | 199/1000 [04:05<16:08,  1.21s/it]
Epoch Progress:  20%|████▊                   | 200/1000 [04:06<16:07,  1.21s/it]Dataset Size: 1%, Epoch: 201, Train Loss: 9.143963694572449, Val Loss: 10.649404798235212

Epoch Progress:  20%|████▊                   | 201/1000 [04:07<16:05,  1.21s/it]
Epoch Progress:  20%|████▊                   | 202/1000 [04:09<16:04,  1.21s/it]
Epoch Progress:  20%|████▊                   | 203/1000 [04:10<16:04,  1.21s/it]
Epoch Progress:  20%|████▉                   | 204/1000 [04:11<16:02,  1.21s/it]
Epoch Progress:  20%|████▉                   | 205/1000 [04:12<16:03,  1.21s/it]
Epoch Progress:  21%|████▉                   | 206/1000 [04:14<16:01,  1.21s/it]
Epoch Progress:  21%|████▉                   | 207/1000 [04:15<16:00,  1.21s/it]
Epoch Progress:  21%|████▉                   | 208/1000 [04:16<15:55,  1.21s/it]
Epoch Progress:  21%|█████                   | 209/1000 [04:17<15:52,  1.20s/it]
Epoch Progress:  21%|█████                   | 210/1000 [04:18<15:51,  1.20s/it]Dataset Size: 1%, Epoch: 211, Train Loss: 9.044595956802368, Val Loss: 10.67257191298844

Epoch Progress:  21%|█████                   | 211/1000 [04:20<15:52,  1.21s/it]
Epoch Progress:  21%|█████                   | 212/1000 [04:21<15:51,  1.21s/it]
Epoch Progress:  21%|█████                   | 213/1000 [04:22<15:50,  1.21s/it]
Epoch Progress:  21%|█████▏                  | 214/1000 [04:23<15:48,  1.21s/it]
Epoch Progress:  22%|█████▏                  | 215/1000 [04:24<15:48,  1.21s/it]
Epoch Progress:  22%|█████▏                  | 216/1000 [04:26<15:50,  1.21s/it]
Epoch Progress:  22%|█████▏                  | 217/1000 [04:27<15:53,  1.22s/it]
Epoch Progress:  22%|█████▏                  | 218/1000 [04:28<15:51,  1.22s/it]
Epoch Progress:  22%|█████▎                  | 219/1000 [04:29<15:51,  1.22s/it]
Epoch Progress:  22%|█████▎                  | 220/1000 [04:30<15:52,  1.22s/it]Dataset Size: 1%, Epoch: 221, Train Loss: 8.93281090259552, Val Loss: 10.685984351418234

Epoch Progress:  22%|█████▎                  | 221/1000 [04:32<15:46,  1.21s/it]
Epoch Progress:  22%|█████▎                  | 222/1000 [04:33<15:42,  1.21s/it]
Epoch Progress:  22%|█████▎                  | 223/1000 [04:34<15:39,  1.21s/it]
Epoch Progress:  22%|█████▍                  | 224/1000 [04:35<15:39,  1.21s/it]
Epoch Progress:  22%|█████▍                  | 225/1000 [04:37<15:37,  1.21s/it]
Epoch Progress:  23%|█████▍                  | 226/1000 [04:38<15:35,  1.21s/it]
Epoch Progress:  23%|█████▍                  | 227/1000 [04:39<15:32,  1.21s/it]
Epoch Progress:  23%|█████▍                  | 228/1000 [04:40<15:32,  1.21s/it]
Epoch Progress:  23%|█████▍                  | 229/1000 [04:41<15:30,  1.21s/it]
Epoch Progress:  23%|█████▌                  | 230/1000 [04:43<15:29,  1.21s/it]Dataset Size: 1%, Epoch: 231, Train Loss: 8.905247569084167, Val Loss: 10.7085623679223

Epoch Progress:  23%|█████▌                  | 231/1000 [04:44<15:28,  1.21s/it]
Epoch Progress:  23%|█████▌                  | 232/1000 [04:45<15:27,  1.21s/it]
Epoch Progress:  23%|█████▌                  | 233/1000 [04:46<15:25,  1.21s/it]
Epoch Progress:  23%|█████▌                  | 234/1000 [04:47<15:26,  1.21s/it]
Epoch Progress:  24%|█████▋                  | 235/1000 [04:49<15:24,  1.21s/it]
Epoch Progress:  24%|█████▋                  | 236/1000 [04:50<15:24,  1.21s/it]
Epoch Progress:  24%|█████▋                  | 237/1000 [04:51<15:21,  1.21s/it]
Epoch Progress:  24%|█████▋                  | 238/1000 [04:52<15:19,  1.21s/it]
Epoch Progress:  24%|█████▋                  | 239/1000 [04:53<15:17,  1.21s/it]
Epoch Progress:  24%|█████▊                  | 240/1000 [04:55<15:15,  1.20s/it]Dataset Size: 1%, Epoch: 241, Train Loss: 8.803794026374817, Val Loss: 10.735566386928806

Epoch Progress:  24%|█████▊                  | 241/1000 [04:56<15:14,  1.21s/it]
Epoch Progress:  24%|█████▊                  | 242/1000 [04:57<15:14,  1.21s/it]
Epoch Progress:  24%|█████▊                  | 243/1000 [04:58<15:16,  1.21s/it]
Epoch Progress:  24%|█████▊                  | 244/1000 [04:59<15:16,  1.21s/it]
Epoch Progress:  24%|█████▉                  | 245/1000 [05:01<15:14,  1.21s/it]
Epoch Progress:  25%|█████▉                  | 246/1000 [05:02<15:12,  1.21s/it]
Epoch Progress:  25%|█████▉                  | 247/1000 [05:03<15:10,  1.21s/it]
Epoch Progress:  25%|█████▉                  | 248/1000 [05:04<15:09,  1.21s/it]
Epoch Progress:  25%|█████▉                  | 249/1000 [05:06<15:08,  1.21s/it]
Epoch Progress:  25%|██████                  | 250/1000 [05:07<15:04,  1.21s/it]Dataset Size: 1%, Epoch: 251, Train Loss: 8.751823663711548, Val Loss: 10.753334900001427

Epoch Progress:  25%|██████                  | 251/1000 [05:08<15:04,  1.21s/it]
Epoch Progress:  25%|██████                  | 252/1000 [05:09<15:05,  1.21s/it]
Epoch Progress:  25%|██████                  | 253/1000 [05:10<15:09,  1.22s/it]
Epoch Progress:  25%|██████                  | 254/1000 [05:12<15:07,  1.22s/it]
Epoch Progress:  26%|██████                  | 255/1000 [05:13<15:05,  1.21s/it]
Epoch Progress:  26%|██████▏                 | 256/1000 [05:14<15:06,  1.22s/it]
Epoch Progress:  26%|██████▏                 | 257/1000 [05:15<15:04,  1.22s/it]
Epoch Progress:  26%|██████▏                 | 258/1000 [05:16<15:01,  1.21s/it]
Epoch Progress:  26%|██████▏                 | 259/1000 [05:18<14:56,  1.21s/it]
Epoch Progress:  26%|██████▏                 | 260/1000 [05:19<14:54,  1.21s/it]Dataset Size: 1%, Epoch: 261, Train Loss: 8.733372807502747, Val Loss: 10.753267498759481

Epoch Progress:  26%|██████▎                 | 261/1000 [05:20<14:55,  1.21s/it]
Epoch Progress:  26%|██████▎                 | 262/1000 [05:21<14:53,  1.21s/it]
Epoch Progress:  26%|██████▎                 | 263/1000 [05:22<14:52,  1.21s/it]
Epoch Progress:  26%|██████▎                 | 264/1000 [05:24<14:52,  1.21s/it]
Epoch Progress:  26%|██████▎                 | 265/1000 [05:25<14:51,  1.21s/it]
Epoch Progress:  27%|██████▍                 | 266/1000 [05:26<14:47,  1.21s/it]
Epoch Progress:  27%|██████▍                 | 267/1000 [05:27<14:45,  1.21s/it]
Epoch Progress:  27%|██████▍                 | 268/1000 [05:29<14:43,  1.21s/it]
Epoch Progress:  27%|██████▍                 | 269/1000 [05:30<14:52,  1.22s/it]
Epoch Progress:  27%|██████▍                 | 270/1000 [05:31<14:47,  1.22s/it]Dataset Size: 1%, Epoch: 271, Train Loss: 8.684458613395691, Val Loss: 10.769171380377434

Epoch Progress:  27%|██████▌                 | 271/1000 [05:32<14:44,  1.21s/it]
Epoch Progress:  27%|██████▌                 | 272/1000 [05:33<14:43,  1.21s/it]
Epoch Progress:  27%|██████▌                 | 273/1000 [05:35<14:40,  1.21s/it]
Epoch Progress:  27%|██████▌                 | 274/1000 [05:36<14:40,  1.21s/it]
Epoch Progress:  28%|██████▌                 | 275/1000 [05:37<14:40,  1.21s/it]
Epoch Progress:  28%|██████▌                 | 276/1000 [05:38<14:39,  1.21s/it]
Epoch Progress:  28%|██████▋                 | 277/1000 [05:39<14:41,  1.22s/it]
Epoch Progress:  28%|██████▋                 | 278/1000 [05:41<14:37,  1.22s/it]
Epoch Progress:  28%|██████▋                 | 279/1000 [05:42<14:36,  1.22s/it]
Epoch Progress:  28%|██████▋                 | 280/1000 [05:43<14:35,  1.22s/it]Dataset Size: 1%, Epoch: 281, Train Loss: 8.646939158439636, Val Loss: 10.780316129907385

Epoch Progress:  28%|██████▋                 | 281/1000 [05:44<14:32,  1.21s/it]
Epoch Progress:  28%|██████▊                 | 282/1000 [05:46<14:28,  1.21s/it]
Epoch Progress:  28%|██████▊                 | 283/1000 [05:47<14:26,  1.21s/it]
Epoch Progress:  28%|██████▊                 | 284/1000 [05:48<14:27,  1.21s/it]
Epoch Progress:  28%|██████▊                 | 285/1000 [05:49<14:23,  1.21s/it]
Epoch Progress:  29%|██████▊                 | 286/1000 [05:50<14:22,  1.21s/it]
Epoch Progress:  29%|██████▉                 | 287/1000 [05:52<14:21,  1.21s/it]
Epoch Progress:  29%|██████▉                 | 288/1000 [05:53<14:22,  1.21s/it]
Epoch Progress:  29%|██████▉                 | 289/1000 [05:54<14:20,  1.21s/it]
Epoch Progress:  29%|██████▉                 | 290/1000 [05:55<14:20,  1.21s/it]Dataset Size: 1%, Epoch: 291, Train Loss: 8.610373258590698, Val Loss: 10.796864224718762

Epoch Progress:  29%|██████▉                 | 291/1000 [05:56<14:16,  1.21s/it]
Epoch Progress:  29%|███████                 | 292/1000 [05:58<14:14,  1.21s/it]
Epoch Progress:  29%|███████                 | 293/1000 [05:59<14:12,  1.21s/it]
Epoch Progress:  29%|███████                 | 294/1000 [06:00<14:13,  1.21s/it]
Epoch Progress:  30%|███████                 | 295/1000 [06:01<14:15,  1.21s/it]
Epoch Progress:  30%|███████                 | 296/1000 [06:02<14:11,  1.21s/it]
Epoch Progress:  30%|███████▏                | 297/1000 [06:04<14:10,  1.21s/it]
Epoch Progress:  30%|███████▏                | 298/1000 [06:05<14:06,  1.21s/it]
Epoch Progress:  30%|███████▏                | 299/1000 [06:06<14:05,  1.21s/it]
Epoch Progress:  30%|███████▏                | 300/1000 [06:07<14:03,  1.21s/it]Dataset Size: 1%, Epoch: 301, Train Loss: 8.585675835609436, Val Loss: 10.80576077993814

Epoch Progress:  30%|███████▏                | 301/1000 [06:08<14:01,  1.20s/it]
Epoch Progress:  30%|███████▏                | 302/1000 [06:10<14:02,  1.21s/it]
Epoch Progress:  30%|███████▎                | 303/1000 [06:11<13:59,  1.20s/it]
Epoch Progress:  30%|███████▎                | 304/1000 [06:12<13:58,  1.20s/it]
Epoch Progress:  30%|███████▎                | 305/1000 [06:13<14:00,  1.21s/it]
Epoch Progress:  31%|███████▎                | 306/1000 [06:15<13:59,  1.21s/it]
Epoch Progress:  31%|███████▎                | 307/1000 [06:16<13:55,  1.21s/it]
Epoch Progress:  31%|███████▍                | 308/1000 [06:17<13:53,  1.20s/it]
Epoch Progress:  31%|███████▍                | 309/1000 [06:18<13:51,  1.20s/it]
Epoch Progress:  31%|███████▍                | 310/1000 [06:19<13:51,  1.21s/it]Dataset Size: 1%, Epoch: 311, Train Loss: 8.560840129852295, Val Loss: 10.818787599538828

Epoch Progress:  31%|███████▍                | 311/1000 [06:21<13:49,  1.20s/it]
Epoch Progress:  31%|███████▍                | 312/1000 [06:22<13:46,  1.20s/it]
Epoch Progress:  31%|███████▌                | 313/1000 [06:23<13:44,  1.20s/it]
Epoch Progress:  31%|███████▌                | 314/1000 [06:24<13:46,  1.20s/it]
Epoch Progress:  32%|███████▌                | 315/1000 [06:25<13:42,  1.20s/it]
Epoch Progress:  32%|███████▌                | 316/1000 [06:27<13:40,  1.20s/it]
Epoch Progress:  32%|███████▌                | 317/1000 [06:28<13:38,  1.20s/it]
Epoch Progress:  32%|███████▋                | 318/1000 [06:29<13:36,  1.20s/it]
Epoch Progress:  32%|███████▋                | 319/1000 [06:30<13:38,  1.20s/it]
Epoch Progress:  32%|███████▋                | 320/1000 [06:31<13:35,  1.20s/it]Dataset Size: 1%, Epoch: 321, Train Loss: 8.550241708755493, Val Loss: 10.834012390731218

Epoch Progress:  32%|███████▋                | 321/1000 [06:33<13:38,  1.21s/it]
Epoch Progress:  32%|███████▋                | 322/1000 [06:34<13:37,  1.21s/it]
Epoch Progress:  32%|███████▊                | 323/1000 [06:35<13:34,  1.20s/it]
Epoch Progress:  32%|███████▊                | 324/1000 [06:36<13:31,  1.20s/it]
Epoch Progress:  32%|███████▊                | 325/1000 [06:37<13:29,  1.20s/it]
Epoch Progress:  33%|███████▊                | 326/1000 [06:39<13:26,  1.20s/it]
Epoch Progress:  33%|███████▊                | 327/1000 [06:40<13:27,  1.20s/it]
Epoch Progress:  33%|███████▊                | 328/1000 [06:41<13:25,  1.20s/it]
Epoch Progress:  33%|███████▉                | 329/1000 [06:42<13:23,  1.20s/it]
Epoch Progress:  33%|███████▉                | 330/1000 [06:43<13:20,  1.20s/it]Dataset Size: 1%, Epoch: 331, Train Loss: 8.54044234752655, Val Loss: 10.839081516513577

Epoch Progress:  33%|███████▉                | 331/1000 [06:45<13:23,  1.20s/it]
Epoch Progress:  33%|███████▉                | 332/1000 [06:46<13:21,  1.20s/it]
Epoch Progress:  33%|███████▉                | 333/1000 [06:47<13:20,  1.20s/it]
Epoch Progress:  33%|████████                | 334/1000 [06:48<13:21,  1.20s/it]
Epoch Progress:  34%|████████                | 335/1000 [06:49<13:19,  1.20s/it]
Epoch Progress:  34%|████████                | 336/1000 [06:51<13:18,  1.20s/it]
Epoch Progress:  34%|████████                | 337/1000 [06:52<13:16,  1.20s/it]
Epoch Progress:  34%|████████                | 338/1000 [06:53<13:14,  1.20s/it]
Epoch Progress:  34%|████████▏               | 339/1000 [06:54<13:12,  1.20s/it]
Epoch Progress:  34%|████████▏               | 340/1000 [06:55<13:10,  1.20s/it]Dataset Size: 1%, Epoch: 341, Train Loss: 8.50912320613861, Val Loss: 10.851213455200195

Epoch Progress:  34%|████████▏               | 341/1000 [06:57<13:08,  1.20s/it]
Epoch Progress:  34%|████████▏               | 342/1000 [06:58<13:06,  1.20s/it]
Epoch Progress:  34%|████████▏               | 343/1000 [06:59<13:07,  1.20s/it]
Epoch Progress:  34%|████████▎               | 344/1000 [07:00<13:10,  1.21s/it]
Epoch Progress:  34%|████████▎               | 345/1000 [07:01<13:06,  1.20s/it]
Epoch Progress:  35%|████████▎               | 346/1000 [07:03<13:03,  1.20s/it]
Epoch Progress:  35%|████████▎               | 347/1000 [07:04<13:04,  1.20s/it]
Epoch Progress:  35%|████████▎               | 348/1000 [07:05<13:06,  1.21s/it]
Epoch Progress:  35%|████████▍               | 349/1000 [07:06<13:04,  1.20s/it]
Epoch Progress:  35%|████████▍               | 350/1000 [07:07<13:02,  1.20s/it]Dataset Size: 1%, Epoch: 351, Train Loss: 8.49186372756958, Val Loss: 10.855200532194855

Epoch Progress:  35%|████████▍               | 351/1000 [07:09<13:01,  1.20s/it]
Epoch Progress:  35%|████████▍               | 352/1000 [07:10<13:00,  1.20s/it]
Epoch Progress:  35%|████████▍               | 353/1000 [07:11<12:56,  1.20s/it]
Epoch Progress:  35%|████████▍               | 354/1000 [07:12<12:54,  1.20s/it]
Epoch Progress:  36%|████████▌               | 355/1000 [07:13<12:54,  1.20s/it]
Epoch Progress:  36%|████████▌               | 356/1000 [07:15<12:54,  1.20s/it]
Epoch Progress:  36%|████████▌               | 357/1000 [07:16<12:53,  1.20s/it]
Epoch Progress:  36%|████████▌               | 358/1000 [07:17<12:52,  1.20s/it]
Epoch Progress:  36%|████████▌               | 359/1000 [07:18<12:51,  1.20s/it]
Epoch Progress:  36%|████████▋               | 360/1000 [07:19<12:57,  1.21s/it]Dataset Size: 1%, Epoch: 361, Train Loss: 8.456541299819946, Val Loss: 10.867017151473405

Epoch Progress:  36%|████████▋               | 361/1000 [07:21<12:59,  1.22s/it]
Epoch Progress:  36%|████████▋               | 362/1000 [07:22<12:56,  1.22s/it]
Epoch Progress:  36%|████████▋               | 363/1000 [07:23<12:53,  1.21s/it]
Epoch Progress:  36%|████████▋               | 364/1000 [07:24<12:53,  1.22s/it]
Epoch Progress:  36%|████████▊               | 365/1000 [07:26<12:54,  1.22s/it]
Epoch Progress:  37%|████████▊               | 366/1000 [07:27<12:51,  1.22s/it]
Epoch Progress:  37%|████████▊               | 367/1000 [07:28<12:48,  1.21s/it]
Epoch Progress:  37%|████████▊               | 368/1000 [07:29<12:45,  1.21s/it]
Epoch Progress:  37%|████████▊               | 369/1000 [07:30<12:44,  1.21s/it]
Epoch Progress:  37%|████████▉               | 370/1000 [07:32<12:42,  1.21s/it]Dataset Size: 1%, Epoch: 371, Train Loss: 8.425075769424438, Val Loss: 10.875123643255852

Epoch Progress:  37%|████████▉               | 371/1000 [07:33<12:39,  1.21s/it]
Epoch Progress:  37%|████████▉               | 372/1000 [07:34<12:37,  1.21s/it]
Epoch Progress:  37%|████████▉               | 373/1000 [07:35<12:35,  1.21s/it]
Epoch Progress:  37%|████████▉               | 374/1000 [07:36<12:39,  1.21s/it]
Epoch Progress:  38%|█████████               | 375/1000 [07:38<12:37,  1.21s/it]
Epoch Progress:  38%|█████████               | 376/1000 [07:39<12:34,  1.21s/it]
Epoch Progress:  38%|█████████               | 377/1000 [07:40<12:34,  1.21s/it]
Epoch Progress:  38%|█████████               | 378/1000 [07:41<12:32,  1.21s/it]
Epoch Progress:  38%|█████████               | 379/1000 [07:42<12:30,  1.21s/it]
Epoch Progress:  38%|█████████               | 380/1000 [07:44<12:29,  1.21s/it]Dataset Size: 1%, Epoch: 381, Train Loss: 8.430347204208374, Val Loss: 10.877032007489886

Epoch Progress:  38%|█████████▏              | 381/1000 [07:45<12:26,  1.21s/it]
Epoch Progress:  38%|█████████▏              | 382/1000 [07:46<12:23,  1.20s/it]
Epoch Progress:  38%|█████████▏              | 383/1000 [07:47<12:21,  1.20s/it]
Epoch Progress:  38%|█████████▏              | 384/1000 [07:48<12:19,  1.20s/it]
Epoch Progress:  38%|█████████▏              | 385/1000 [07:50<12:20,  1.20s/it]
Epoch Progress:  39%|█████████▎              | 386/1000 [07:51<12:18,  1.20s/it]
Epoch Progress:  39%|█████████▎              | 387/1000 [07:52<12:20,  1.21s/it]
Epoch Progress:  39%|█████████▎              | 388/1000 [07:53<12:17,  1.21s/it]
Epoch Progress:  39%|█████████▎              | 389/1000 [07:54<12:16,  1.21s/it]
Epoch Progress:  39%|█████████▎              | 390/1000 [07:56<12:14,  1.20s/it]Dataset Size: 1%, Epoch: 391, Train Loss: 8.438124179840088, Val Loss: 10.886137714633694

Epoch Progress:  39%|█████████▍              | 391/1000 [07:57<12:13,  1.20s/it]
Epoch Progress:  39%|█████████▍              | 392/1000 [07:58<12:10,  1.20s/it]
Epoch Progress:  39%|█████████▍              | 393/1000 [07:59<12:10,  1.20s/it]
Epoch Progress:  39%|█████████▍              | 394/1000 [08:01<12:11,  1.21s/it]
Epoch Progress:  40%|█████████▍              | 395/1000 [08:02<12:08,  1.20s/it]
Epoch Progress:  40%|█████████▌              | 396/1000 [08:03<12:09,  1.21s/it]
Epoch Progress:  40%|█████████▌              | 397/1000 [08:04<12:06,  1.21s/it]
Epoch Progress:  40%|█████████▌              | 398/1000 [08:05<12:03,  1.20s/it]
Epoch Progress:  40%|█████████▌              | 399/1000 [08:07<12:02,  1.20s/it]
Epoch Progress:  40%|█████████▌              | 400/1000 [08:08<12:06,  1.21s/it]Dataset Size: 1%, Epoch: 401, Train Loss: 8.442508935928345, Val Loss: 10.901009373850636

Epoch Progress:  40%|█████████▌              | 401/1000 [08:09<12:05,  1.21s/it]
Epoch Progress:  40%|█████████▋              | 402/1000 [08:10<12:06,  1.21s/it]
Epoch Progress:  40%|█████████▋              | 403/1000 [08:11<12:03,  1.21s/it]
Epoch Progress:  40%|█████████▋              | 404/1000 [08:13<12:00,  1.21s/it]
Epoch Progress:  40%|█████████▋              | 405/1000 [08:14<11:59,  1.21s/it]
Epoch Progress:  41%|█████████▋              | 406/1000 [08:15<11:55,  1.21s/it]
Epoch Progress:  41%|█████████▊              | 407/1000 [08:16<11:53,  1.20s/it]
Epoch Progress:  41%|█████████▊              | 408/1000 [08:17<11:50,  1.20s/it]
Epoch Progress:  41%|█████████▊              | 409/1000 [08:19<11:49,  1.20s/it]
Epoch Progress:  41%|█████████▊              | 410/1000 [08:20<11:51,  1.21s/it]Dataset Size: 1%, Epoch: 411, Train Loss: 8.414026975631714, Val Loss: 10.909600579893434

Epoch Progress:  41%|█████████▊              | 411/1000 [08:21<11:48,  1.20s/it]
Epoch Progress:  41%|█████████▉              | 412/1000 [08:22<11:47,  1.20s/it]
Epoch Progress:  41%|█████████▉              | 413/1000 [08:23<11:49,  1.21s/it]
Epoch Progress:  41%|█████████▉              | 414/1000 [08:25<11:54,  1.22s/it]
Epoch Progress:  42%|█████████▉              | 415/1000 [08:26<12:02,  1.24s/it]
Epoch Progress:  42%|█████████▉              | 416/1000 [08:27<12:04,  1.24s/it]
Epoch Progress:  42%|██████████              | 417/1000 [08:28<11:58,  1.23s/it]
Epoch Progress:  42%|██████████              | 418/1000 [08:30<11:56,  1.23s/it]
Epoch Progress:  42%|██████████              | 419/1000 [08:31<11:51,  1.22s/it]
Epoch Progress:  42%|██████████              | 420/1000 [08:32<11:53,  1.23s/it]Dataset Size: 1%, Epoch: 421, Train Loss: 8.37319016456604, Val Loss: 10.913399312403294

Epoch Progress:  42%|██████████              | 421/1000 [08:33<11:57,  1.24s/it]
Epoch Progress:  42%|██████████▏             | 422/1000 [08:35<11:55,  1.24s/it]
Epoch Progress:  42%|██████████▏             | 423/1000 [08:36<11:49,  1.23s/it]
Epoch Progress:  42%|██████████▏             | 424/1000 [08:37<11:44,  1.22s/it]
Epoch Progress:  42%|██████████▏             | 425/1000 [08:38<11:40,  1.22s/it]
Epoch Progress:  43%|██████████▏             | 426/1000 [08:39<11:44,  1.23s/it]
Epoch Progress:  43%|██████████▏             | 427/1000 [08:41<11:39,  1.22s/it]
Epoch Progress:  43%|██████████▎             | 428/1000 [08:42<11:34,  1.21s/it]
Epoch Progress:  43%|██████████▎             | 429/1000 [08:43<11:31,  1.21s/it]
Epoch Progress:  43%|██████████▎             | 430/1000 [08:44<11:31,  1.21s/it]Dataset Size: 1%, Epoch: 431, Train Loss: 8.362187147140503, Val Loss: 10.922849432214514

Epoch Progress:  43%|██████████▎             | 431/1000 [08:46<11:29,  1.21s/it]
Epoch Progress:  43%|██████████▎             | 432/1000 [08:47<11:27,  1.21s/it]
Epoch Progress:  43%|██████████▍             | 433/1000 [08:48<11:25,  1.21s/it]
Epoch Progress:  43%|██████████▍             | 434/1000 [08:49<11:23,  1.21s/it]
Epoch Progress:  44%|██████████▍             | 435/1000 [08:50<11:24,  1.21s/it]
Epoch Progress:  44%|██████████▍             | 436/1000 [08:52<11:23,  1.21s/it]
Epoch Progress:  44%|██████████▍             | 437/1000 [08:53<11:21,  1.21s/it]
Epoch Progress:  44%|██████████▌             | 438/1000 [08:54<11:19,  1.21s/it]
Epoch Progress:  44%|██████████▌             | 439/1000 [08:55<11:17,  1.21s/it]
Epoch Progress:  44%|██████████▌             | 440/1000 [08:56<11:17,  1.21s/it]Dataset Size: 1%, Epoch: 441, Train Loss: 8.30459976196289, Val Loss: 10.929863533416352

Epoch Progress:  44%|██████████▌             | 441/1000 [08:58<11:15,  1.21s/it]
Epoch Progress:  44%|██████████▌             | 442/1000 [08:59<11:16,  1.21s/it]
Epoch Progress:  44%|██████████▋             | 443/1000 [09:00<11:17,  1.22s/it]
Epoch Progress:  44%|██████████▋             | 444/1000 [09:01<11:13,  1.21s/it]
Epoch Progress:  44%|██████████▋             | 445/1000 [09:02<11:11,  1.21s/it]
Epoch Progress:  45%|██████████▋             | 446/1000 [09:04<11:10,  1.21s/it]
Epoch Progress:  45%|██████████▋             | 447/1000 [09:05<11:09,  1.21s/it]
Epoch Progress:  45%|██████████▊             | 448/1000 [09:06<11:07,  1.21s/it]
Epoch Progress:  45%|██████████▊             | 449/1000 [09:07<11:05,  1.21s/it]
Epoch Progress:  45%|██████████▊             | 450/1000 [09:08<11:03,  1.21s/it]Dataset Size: 1%, Epoch: 451, Train Loss: 8.357282519340515, Val Loss: 10.945005837973062

Epoch Progress:  45%|██████████▊             | 451/1000 [09:10<11:03,  1.21s/it]
Epoch Progress:  45%|██████████▊             | 452/1000 [09:11<11:06,  1.22s/it]
Epoch Progress:  45%|██████████▊             | 453/1000 [09:12<11:04,  1.22s/it]
Epoch Progress:  45%|██████████▉             | 454/1000 [09:13<11:01,  1.21s/it]
Epoch Progress:  46%|██████████▉             | 455/1000 [09:15<10:58,  1.21s/it]
Epoch Progress:  46%|██████████▉             | 456/1000 [09:16<10:55,  1.20s/it]
Epoch Progress:  46%|██████████▉             | 457/1000 [09:17<10:52,  1.20s/it]
Epoch Progress:  46%|██████████▉             | 458/1000 [09:18<10:51,  1.20s/it]
Epoch Progress:  46%|███████████             | 459/1000 [09:19<10:51,  1.20s/it]
Epoch Progress:  46%|███████████             | 460/1000 [09:21<10:51,  1.21s/it]Dataset Size: 1%, Epoch: 461, Train Loss: 8.347904086112976, Val Loss: 10.948504745186149

Epoch Progress:  46%|███████████             | 461/1000 [09:22<10:50,  1.21s/it]
Epoch Progress:  46%|███████████             | 462/1000 [09:23<10:47,  1.20s/it]
Epoch Progress:  46%|███████████             | 463/1000 [09:24<10:49,  1.21s/it]
Epoch Progress:  46%|███████████▏            | 464/1000 [09:25<10:48,  1.21s/it]
Epoch Progress:  46%|███████████▏            | 465/1000 [09:27<10:45,  1.21s/it]
Epoch Progress:  47%|███████████▏            | 466/1000 [09:28<10:43,  1.20s/it]
Epoch Progress:  47%|███████████▏            | 467/1000 [09:29<10:42,  1.21s/it]
Epoch Progress:  47%|███████████▏            | 468/1000 [09:30<10:44,  1.21s/it]
Epoch Progress:  47%|███████████▎            | 469/1000 [09:31<10:42,  1.21s/it]
Epoch Progress:  47%|███████████▎            | 470/1000 [09:33<10:39,  1.21s/it]Dataset Size: 1%, Epoch: 471, Train Loss: 8.302719712257385, Val Loss: 10.946986879621234

Epoch Progress:  47%|███████████▎            | 471/1000 [09:34<10:37,  1.20s/it]
Epoch Progress:  47%|███████████▎            | 472/1000 [09:35<10:34,  1.20s/it]
Epoch Progress:  47%|███████████▎            | 473/1000 [09:36<10:34,  1.20s/it]
Epoch Progress:  47%|███████████▍            | 474/1000 [09:37<10:32,  1.20s/it]
Epoch Progress:  48%|███████████▍            | 475/1000 [09:39<10:31,  1.20s/it]
Epoch Progress:  48%|███████████▍            | 476/1000 [09:40<10:32,  1.21s/it]
Epoch Progress:  48%|███████████▍            | 477/1000 [09:41<10:30,  1.21s/it]
Epoch Progress:  48%|███████████▍            | 478/1000 [09:42<10:31,  1.21s/it]
Epoch Progress:  48%|███████████▍            | 479/1000 [09:43<10:29,  1.21s/it]
Epoch Progress:  48%|███████████▌            | 480/1000 [09:45<10:27,  1.21s/it]Dataset Size: 1%, Epoch: 481, Train Loss: 8.337509036064148, Val Loss: 10.951965691207292

Epoch Progress:  48%|███████████▌            | 481/1000 [09:46<10:24,  1.20s/it]
Epoch Progress:  48%|███████████▌            | 482/1000 [09:47<10:25,  1.21s/it]
Epoch Progress:  48%|███████████▌            | 483/1000 [09:48<10:22,  1.20s/it]
Epoch Progress:  48%|███████████▌            | 484/1000 [09:50<10:23,  1.21s/it]
Epoch Progress:  48%|███████████▋            | 485/1000 [09:51<10:22,  1.21s/it]
Epoch Progress:  49%|███████████▋            | 486/1000 [09:52<10:21,  1.21s/it]
Epoch Progress:  49%|███████████▋            | 487/1000 [09:53<10:18,  1.21s/it]
Epoch Progress:  49%|███████████▋            | 488/1000 [09:54<10:17,  1.21s/it]
Epoch Progress:  49%|███████████▋            | 489/1000 [09:56<10:15,  1.21s/it]
Epoch Progress:  49%|███████████▊            | 490/1000 [09:57<10:14,  1.20s/it]Dataset Size: 1%, Epoch: 491, Train Loss: 8.324230551719666, Val Loss: 10.956038648431951

Epoch Progress:  49%|███████████▊            | 491/1000 [09:58<10:14,  1.21s/it]
Epoch Progress:  49%|███████████▊            | 492/1000 [09:59<10:18,  1.22s/it]
Epoch Progress:  49%|███████████▊            | 493/1000 [10:00<10:21,  1.23s/it]
Epoch Progress:  49%|███████████▊            | 494/1000 [10:02<10:18,  1.22s/it]
Epoch Progress:  50%|███████████▉            | 495/1000 [10:03<10:17,  1.22s/it]
Epoch Progress:  50%|███████████▉            | 496/1000 [10:04<10:12,  1.22s/it]
Epoch Progress:  50%|███████████▉            | 497/1000 [10:05<10:14,  1.22s/it]
Epoch Progress:  50%|███████████▉            | 498/1000 [10:07<10:17,  1.23s/it]
Epoch Progress:  50%|███████████▉            | 499/1000 [10:08<10:16,  1.23s/it]
Epoch Progress:  50%|████████████            | 500/1000 [10:09<10:12,  1.22s/it]Dataset Size: 1%, Epoch: 501, Train Loss: 8.358047008514404, Val Loss: 10.964018388227982

Epoch Progress:  50%|████████████            | 501/1000 [10:10<10:11,  1.23s/it]
Epoch Progress:  50%|████████████            | 502/1000 [10:11<10:11,  1.23s/it]
Epoch Progress:  50%|████████████            | 503/1000 [10:13<10:08,  1.22s/it]
Epoch Progress:  50%|████████████            | 504/1000 [10:14<10:11,  1.23s/it]
Epoch Progress:  50%|████████████            | 505/1000 [10:15<10:08,  1.23s/it]
Epoch Progress:  51%|████████████▏           | 506/1000 [10:16<10:05,  1.23s/it]
Epoch Progress:  51%|████████████▏           | 507/1000 [10:18<10:01,  1.22s/it]
Epoch Progress:  51%|████████████▏           | 508/1000 [10:19<09:58,  1.22s/it]
Epoch Progress:  51%|████████████▏           | 509/1000 [10:20<09:58,  1.22s/it]
Epoch Progress:  51%|████████████▏           | 510/1000 [10:21<09:57,  1.22s/it]Dataset Size: 1%, Epoch: 511, Train Loss: 8.31609034538269, Val Loss: 10.968491195084212

Epoch Progress:  51%|████████████▎           | 511/1000 [10:22<09:54,  1.22s/it]
Epoch Progress:  51%|████████████▎           | 512/1000 [10:24<09:53,  1.22s/it]
Epoch Progress:  51%|████████████▎           | 513/1000 [10:25<09:52,  1.22s/it]
Epoch Progress:  51%|████████████▎           | 514/1000 [10:26<09:50,  1.22s/it]
Epoch Progress:  52%|████████████▎           | 515/1000 [10:27<09:49,  1.21s/it]
Epoch Progress:  52%|████████████▍           | 516/1000 [10:29<09:46,  1.21s/it]
Epoch Progress:  52%|████████████▍           | 517/1000 [10:30<09:46,  1.21s/it]
Epoch Progress:  52%|████████████▍           | 518/1000 [10:31<09:45,  1.21s/it]
Epoch Progress:  52%|████████████▍           | 519/1000 [10:32<09:43,  1.21s/it]
Epoch Progress:  52%|████████████▍           | 520/1000 [10:33<09:42,  1.21s/it]Dataset Size: 1%, Epoch: 521, Train Loss: 8.314254641532898, Val Loss: 10.969365887827687

Epoch Progress:  52%|████████████▌           | 521/1000 [10:35<09:41,  1.21s/it]
Epoch Progress:  52%|████████████▌           | 522/1000 [10:36<09:42,  1.22s/it]
Epoch Progress:  52%|████████████▌           | 523/1000 [10:37<09:39,  1.21s/it]
Epoch Progress:  52%|████████████▌           | 524/1000 [10:38<09:36,  1.21s/it]
Epoch Progress:  52%|████████████▌           | 525/1000 [10:39<09:36,  1.21s/it]
Epoch Progress:  53%|████████████▌           | 526/1000 [10:41<09:34,  1.21s/it]
Epoch Progress:  53%|████████████▋           | 527/1000 [10:42<09:32,  1.21s/it]
Epoch Progress:  53%|████████████▋           | 528/1000 [10:43<09:30,  1.21s/it]
Epoch Progress:  53%|████████████▋           | 529/1000 [10:44<09:28,  1.21s/it]
Epoch Progress:  53%|████████████▋           | 530/1000 [10:45<09:30,  1.21s/it]Dataset Size: 1%, Epoch: 531, Train Loss: 8.322465777397156, Val Loss: 10.974686610234247

Epoch Progress:  53%|████████████▋           | 531/1000 [10:47<09:29,  1.21s/it]
Epoch Progress:  53%|████████████▊           | 532/1000 [10:48<09:26,  1.21s/it]
Epoch Progress:  53%|████████████▊           | 533/1000 [10:49<09:24,  1.21s/it]
Epoch Progress:  53%|████████████▊           | 534/1000 [10:50<09:23,  1.21s/it]
Epoch Progress:  54%|████████████▊           | 535/1000 [10:52<09:21,  1.21s/it]
Epoch Progress:  54%|████████████▊           | 536/1000 [10:53<09:20,  1.21s/it]
Epoch Progress:  54%|████████████▉           | 537/1000 [10:54<09:19,  1.21s/it]
Epoch Progress:  54%|████████████▉           | 538/1000 [10:55<09:18,  1.21s/it]
Epoch Progress:  54%|████████████▉           | 539/1000 [10:56<09:14,  1.20s/it]
Epoch Progress:  54%|████████████▉           | 540/1000 [10:58<09:13,  1.20s/it]Dataset Size: 1%, Epoch: 541, Train Loss: 8.302797317504883, Val Loss: 10.976706480051016

Epoch Progress:  54%|████████████▉           | 541/1000 [10:59<09:11,  1.20s/it]
Epoch Progress:  54%|█████████████           | 542/1000 [11:00<09:12,  1.21s/it]
Epoch Progress:  54%|█████████████           | 543/1000 [11:01<09:10,  1.20s/it]
Epoch Progress:  54%|█████████████           | 544/1000 [11:02<09:08,  1.20s/it]
Epoch Progress:  55%|█████████████           | 545/1000 [11:04<09:08,  1.21s/it]
Epoch Progress:  55%|█████████████           | 546/1000 [11:05<09:07,  1.21s/it]
Epoch Progress:  55%|█████████████▏          | 547/1000 [11:06<09:05,  1.20s/it]
Epoch Progress:  55%|█████████████▏          | 548/1000 [11:07<09:05,  1.21s/it]
Epoch Progress:  55%|█████████████▏          | 549/1000 [11:08<09:05,  1.21s/it]
Epoch Progress:  55%|█████████████▏          | 550/1000 [11:10<09:07,  1.22s/it]Dataset Size: 1%, Epoch: 551, Train Loss: 8.297847986221313, Val Loss: 10.983609893105246

Epoch Progress:  55%|█████████████▏          | 551/1000 [11:11<09:04,  1.21s/it]
Epoch Progress:  55%|█████████████▏          | 552/1000 [11:12<09:02,  1.21s/it]
Epoch Progress:  55%|█████████████▎          | 553/1000 [11:13<09:00,  1.21s/it]
Epoch Progress:  55%|█████████████▎          | 554/1000 [11:14<08:58,  1.21s/it]
Epoch Progress:  56%|█████████████▎          | 555/1000 [11:16<08:56,  1.21s/it]
Epoch Progress:  56%|█████████████▎          | 556/1000 [11:17<09:00,  1.22s/it]
Epoch Progress:  56%|█████████████▎          | 557/1000 [11:18<09:00,  1.22s/it]
Epoch Progress:  56%|█████████████▍          | 558/1000 [11:19<08:59,  1.22s/it]
Epoch Progress:  56%|█████████████▍          | 559/1000 [11:21<08:58,  1.22s/it]
Epoch Progress:  56%|█████████████▍          | 560/1000 [11:22<08:56,  1.22s/it]Dataset Size: 1%, Epoch: 561, Train Loss: 8.30811870098114, Val Loss: 10.982915197099958

Epoch Progress:  56%|█████████████▍          | 561/1000 [11:23<08:54,  1.22s/it]
Epoch Progress:  56%|█████████████▍          | 562/1000 [11:24<08:58,  1.23s/it]
Epoch Progress:  56%|█████████████▌          | 563/1000 [11:26<09:01,  1.24s/it]
Epoch Progress:  56%|█████████████▌          | 564/1000 [11:27<09:02,  1.24s/it]
Epoch Progress:  56%|█████████████▌          | 565/1000 [11:28<09:00,  1.24s/it]
Epoch Progress:  57%|█████████████▌          | 566/1000 [11:29<08:56,  1.24s/it]
Epoch Progress:  57%|█████████████▌          | 567/1000 [11:30<08:54,  1.23s/it]
Epoch Progress:  57%|█████████████▋          | 568/1000 [11:32<08:52,  1.23s/it]
Epoch Progress:  57%|█████████████▋          | 569/1000 [11:33<08:48,  1.23s/it]
Epoch Progress:  57%|█████████████▋          | 570/1000 [11:34<08:47,  1.23s/it]Dataset Size: 1%, Epoch: 571, Train Loss: 8.305715799331665, Val Loss: 10.98729116266424

Epoch Progress:  57%|█████████████▋          | 571/1000 [11:35<08:51,  1.24s/it]
Epoch Progress:  57%|█████████████▋          | 572/1000 [11:37<08:53,  1.25s/it]
Epoch Progress:  57%|█████████████▊          | 573/1000 [11:38<08:50,  1.24s/it]
Epoch Progress:  57%|█████████████▊          | 574/1000 [11:39<08:45,  1.23s/it]
Epoch Progress:  57%|█████████████▊          | 575/1000 [11:40<08:42,  1.23s/it]
Epoch Progress:  58%|█████████████▊          | 576/1000 [11:42<08:39,  1.23s/it]
Epoch Progress:  58%|█████████████▊          | 577/1000 [11:43<08:36,  1.22s/it]
Epoch Progress:  58%|█████████████▊          | 578/1000 [11:44<08:37,  1.23s/it]
Epoch Progress:  58%|█████████████▉          | 579/1000 [11:45<08:34,  1.22s/it]
Epoch Progress:  58%|█████████████▉          | 580/1000 [11:46<08:32,  1.22s/it]Dataset Size: 1%, Epoch: 581, Train Loss: 8.271140933036804, Val Loss: 10.989670728708242

Epoch Progress:  58%|█████████████▉          | 581/1000 [11:48<08:30,  1.22s/it]
Epoch Progress:  58%|█████████████▉          | 582/1000 [11:49<08:32,  1.23s/it]
Epoch Progress:  58%|█████████████▉          | 583/1000 [11:50<08:32,  1.23s/it]
Epoch Progress:  58%|██████████████          | 584/1000 [11:51<08:29,  1.23s/it]
Epoch Progress:  58%|██████████████          | 585/1000 [11:53<08:31,  1.23s/it]
Epoch Progress:  59%|██████████████          | 586/1000 [11:54<08:28,  1.23s/it]
Epoch Progress:  59%|██████████████          | 587/1000 [11:55<08:25,  1.22s/it]
Epoch Progress:  59%|██████████████          | 588/1000 [11:56<08:25,  1.23s/it]
Epoch Progress:  59%|██████████████▏         | 589/1000 [11:58<08:25,  1.23s/it]
Epoch Progress:  59%|██████████████▏         | 590/1000 [11:59<08:26,  1.24s/it]Dataset Size: 1%, Epoch: 591, Train Loss: 8.267312467098236, Val Loss: 10.992430587867638

Epoch Progress:  59%|██████████████▏         | 591/1000 [12:00<08:26,  1.24s/it]
Epoch Progress:  59%|██████████████▏         | 592/1000 [12:01<08:21,  1.23s/it]
Epoch Progress:  59%|██████████████▏         | 593/1000 [12:02<08:18,  1.23s/it]
Epoch Progress:  59%|██████████████▎         | 594/1000 [12:04<08:19,  1.23s/it]
Epoch Progress:  60%|██████████████▎         | 595/1000 [12:05<08:19,  1.23s/it]
Epoch Progress:  60%|██████████████▎         | 596/1000 [12:06<08:15,  1.23s/it]
Epoch Progress:  60%|██████████████▎         | 597/1000 [12:07<08:13,  1.22s/it]
Epoch Progress:  60%|██████████████▎         | 598/1000 [12:09<08:10,  1.22s/it]
Epoch Progress:  60%|██████████████▍         | 599/1000 [12:10<08:09,  1.22s/it]
Epoch Progress:  60%|██████████████▍         | 600/1000 [12:11<08:07,  1.22s/it]Dataset Size: 1%, Epoch: 601, Train Loss: 8.267751216888428, Val Loss: 10.998798320819805

Epoch Progress:  60%|██████████████▍         | 601/1000 [12:12<08:05,  1.22s/it]
Epoch Progress:  60%|██████████████▍         | 602/1000 [12:13<08:03,  1.21s/it]
Epoch Progress:  60%|██████████████▍         | 603/1000 [12:15<08:04,  1.22s/it]
Epoch Progress:  60%|██████████████▍         | 604/1000 [12:16<08:01,  1.22s/it]
Epoch Progress:  60%|██████████████▌         | 605/1000 [12:17<07:59,  1.21s/it]
Epoch Progress:  61%|██████████████▌         | 606/1000 [12:18<07:57,  1.21s/it]
Epoch Progress:  61%|██████████████▌         | 607/1000 [12:19<07:57,  1.21s/it]
Epoch Progress:  61%|██████████████▌         | 608/1000 [12:21<07:59,  1.22s/it]
Epoch Progress:  61%|██████████████▌         | 609/1000 [12:22<07:57,  1.22s/it]
Epoch Progress:  61%|██████████████▋         | 610/1000 [12:23<07:55,  1.22s/it]Dataset Size: 1%, Epoch: 611, Train Loss: 8.290529727935791, Val Loss: 10.995387003019259

Epoch Progress:  61%|██████████████▋         | 611/1000 [12:24<07:54,  1.22s/it]
Epoch Progress:  61%|██████████████▋         | 612/1000 [12:26<07:52,  1.22s/it]
Epoch Progress:  61%|██████████████▋         | 613/1000 [12:27<07:50,  1.22s/it]
Epoch Progress:  61%|██████████████▋         | 614/1000 [12:28<07:49,  1.22s/it]
Epoch Progress:  62%|██████████████▊         | 615/1000 [12:29<07:48,  1.22s/it]
Epoch Progress:  62%|██████████████▊         | 616/1000 [12:30<07:47,  1.22s/it]
Epoch Progress:  62%|██████████████▊         | 617/1000 [12:32<07:46,  1.22s/it]
Epoch Progress:  62%|██████████████▊         | 618/1000 [12:33<07:45,  1.22s/it]
Epoch Progress:  62%|██████████████▊         | 619/1000 [12:34<07:44,  1.22s/it]
Epoch Progress:  62%|██████████████▉         | 620/1000 [12:35<07:41,  1.22s/it]Dataset Size: 1%, Epoch: 621, Train Loss: 8.327039241790771, Val Loss: 11.000934204497895

Epoch Progress:  62%|██████████████▉         | 621/1000 [12:37<07:41,  1.22s/it]
Epoch Progress:  62%|██████████████▉         | 622/1000 [12:38<07:40,  1.22s/it]
Epoch Progress:  62%|██████████████▉         | 623/1000 [12:39<07:38,  1.22s/it]
Epoch Progress:  62%|██████████████▉         | 624/1000 [12:40<07:38,  1.22s/it]
Epoch Progress:  62%|███████████████         | 625/1000 [12:41<07:35,  1.22s/it]
Epoch Progress:  63%|███████████████         | 626/1000 [12:43<07:34,  1.21s/it]
Epoch Progress:  63%|███████████████         | 627/1000 [12:44<07:32,  1.21s/it]
Epoch Progress:  63%|███████████████         | 628/1000 [12:45<07:30,  1.21s/it]
Epoch Progress:  63%|███████████████         | 629/1000 [12:46<07:29,  1.21s/it]
Epoch Progress:  63%|███████████████         | 630/1000 [12:47<07:28,  1.21s/it]Dataset Size: 1%, Epoch: 631, Train Loss: 8.261033654212952, Val Loss: 11.001025137963232

Epoch Progress:  63%|███████████████▏        | 631/1000 [12:49<07:27,  1.21s/it]
Epoch Progress:  63%|███████████████▏        | 632/1000 [12:50<07:26,  1.21s/it]
Epoch Progress:  63%|███████████████▏        | 633/1000 [12:51<07:24,  1.21s/it]
Epoch Progress:  63%|███████████████▏        | 634/1000 [12:52<07:26,  1.22s/it]
Epoch Progress:  64%|███████████████▏        | 635/1000 [12:54<07:25,  1.22s/it]
Epoch Progress:  64%|███████████████▎        | 636/1000 [12:55<07:24,  1.22s/it]
Epoch Progress:  64%|███████████████▎        | 637/1000 [12:56<07:21,  1.22s/it]
Epoch Progress:  64%|███████████████▎        | 638/1000 [12:57<07:23,  1.22s/it]
Epoch Progress:  64%|███████████████▎        | 639/1000 [12:58<07:25,  1.23s/it]
Epoch Progress:  64%|███████████████▎        | 640/1000 [13:00<07:29,  1.25s/it]Dataset Size: 1%, Epoch: 641, Train Loss: 8.261167407035828, Val Loss: 10.998005582140637

Epoch Progress:  64%|███████████████▍        | 641/1000 [13:01<07:25,  1.24s/it]
Epoch Progress:  64%|███████████████▍        | 642/1000 [13:02<07:21,  1.23s/it]
Epoch Progress:  64%|███████████████▍        | 643/1000 [13:03<07:17,  1.23s/it]
Epoch Progress:  64%|███████████████▍        | 644/1000 [13:05<07:15,  1.22s/it]
Epoch Progress:  64%|███████████████▍        | 645/1000 [13:06<07:13,  1.22s/it]
Epoch Progress:  65%|███████████████▌        | 646/1000 [13:07<07:11,  1.22s/it]
Epoch Progress:  65%|███████████████▌        | 647/1000 [13:08<07:15,  1.23s/it]
Epoch Progress:  65%|███████████████▌        | 648/1000 [13:10<07:18,  1.24s/it]
Epoch Progress:  65%|███████████████▌        | 649/1000 [13:11<07:15,  1.24s/it]
Epoch Progress:  65%|███████████████▌        | 650/1000 [13:12<07:11,  1.23s/it]Dataset Size: 1%, Epoch: 651, Train Loss: 8.262371897697449, Val Loss: 11.002691306077041

Epoch Progress:  65%|███████████████▌        | 651/1000 [13:13<07:08,  1.23s/it]
Epoch Progress:  65%|███████████████▋        | 652/1000 [13:14<07:06,  1.22s/it]
Epoch Progress:  65%|███████████████▋        | 653/1000 [13:16<07:03,  1.22s/it]
Epoch Progress:  65%|███████████████▋        | 654/1000 [13:17<07:01,  1.22s/it]
Epoch Progress:  66%|███████████████▋        | 655/1000 [13:18<06:58,  1.21s/it]
Epoch Progress:  66%|███████████████▋        | 656/1000 [13:19<06:56,  1.21s/it]
Epoch Progress:  66%|███████████████▊        | 657/1000 [13:21<06:57,  1.22s/it]
Epoch Progress:  66%|███████████████▊        | 658/1000 [13:22<06:55,  1.22s/it]
Epoch Progress:  66%|███████████████▊        | 659/1000 [13:23<06:53,  1.21s/it]
Epoch Progress:  66%|███████████████▊        | 660/1000 [13:24<06:56,  1.22s/it]Dataset Size: 1%, Epoch: 661, Train Loss: 8.273440480232239, Val Loss: 11.002964589502904

Epoch Progress:  66%|███████████████▊        | 661/1000 [13:25<06:54,  1.22s/it]
Epoch Progress:  66%|███████████████▉        | 662/1000 [13:27<06:51,  1.22s/it]
Epoch Progress:  66%|███████████████▉        | 663/1000 [13:28<06:49,  1.21s/it]
Epoch Progress:  66%|███████████████▉        | 664/1000 [13:29<06:46,  1.21s/it]
Epoch Progress:  66%|███████████████▉        | 665/1000 [13:30<06:49,  1.22s/it]
Epoch Progress:  67%|███████████████▉        | 666/1000 [13:32<06:50,  1.23s/it]
Epoch Progress:  67%|████████████████        | 667/1000 [13:33<06:51,  1.24s/it]
Epoch Progress:  67%|████████████████        | 668/1000 [13:34<06:50,  1.24s/it]
Epoch Progress:  67%|████████████████        | 669/1000 [13:35<06:46,  1.23s/it]
Epoch Progress:  67%|████████████████        | 670/1000 [13:36<06:45,  1.23s/it]Dataset Size: 1%, Epoch: 671, Train Loss: 8.286663055419922, Val Loss: 11.00156899241658

Epoch Progress:  67%|████████████████        | 671/1000 [13:38<06:47,  1.24s/it]
Epoch Progress:  67%|████████████████▏       | 672/1000 [13:39<06:45,  1.24s/it]
Epoch Progress:  67%|████████████████▏       | 673/1000 [13:40<06:47,  1.25s/it]
Epoch Progress:  67%|████████████████▏       | 674/1000 [13:41<06:44,  1.24s/it]
Epoch Progress:  68%|████████████████▏       | 675/1000 [13:43<06:40,  1.23s/it]
Epoch Progress:  68%|████████████████▏       | 676/1000 [13:44<06:37,  1.23s/it]
Epoch Progress:  68%|████████████████▏       | 677/1000 [13:45<06:34,  1.22s/it]
Epoch Progress:  68%|████████████████▎       | 678/1000 [13:46<06:33,  1.22s/it]
Epoch Progress:  68%|████████████████▎       | 679/1000 [13:48<06:33,  1.23s/it]
Epoch Progress:  68%|████████████████▎       | 680/1000 [13:49<06:31,  1.22s/it]Dataset Size: 1%, Epoch: 681, Train Loss: 8.27730643749237, Val Loss: 11.009009844296939

Epoch Progress:  68%|████████████████▎       | 681/1000 [13:50<06:30,  1.23s/it]
Epoch Progress:  68%|████████████████▎       | 682/1000 [13:51<06:27,  1.22s/it]
Epoch Progress:  68%|████████████████▍       | 683/1000 [13:52<06:26,  1.22s/it]
Epoch Progress:  68%|████████████████▍       | 684/1000 [13:54<06:25,  1.22s/it]
Epoch Progress:  68%|████████████████▍       | 685/1000 [13:55<06:23,  1.22s/it]
Epoch Progress:  69%|████████████████▍       | 686/1000 [13:56<06:25,  1.23s/it]
Epoch Progress:  69%|████████████████▍       | 687/1000 [13:57<06:21,  1.22s/it]
Epoch Progress:  69%|████████████████▌       | 688/1000 [13:59<06:20,  1.22s/it]
Epoch Progress:  69%|████████████████▌       | 689/1000 [14:00<06:19,  1.22s/it]
Epoch Progress:  69%|████████████████▌       | 690/1000 [14:01<06:16,  1.22s/it]Dataset Size: 1%, Epoch: 691, Train Loss: 8.287498414516449, Val Loss: 11.009120742996018

Epoch Progress:  69%|████████████████▌       | 691/1000 [14:02<06:14,  1.21s/it]
Epoch Progress:  69%|████████████████▌       | 692/1000 [14:03<06:13,  1.21s/it]
Epoch Progress:  69%|████████████████▋       | 693/1000 [14:05<06:11,  1.21s/it]
Epoch Progress:  69%|████████████████▋       | 694/1000 [14:06<06:09,  1.21s/it]
Epoch Progress:  70%|████████████████▋       | 695/1000 [14:07<06:08,  1.21s/it]
Epoch Progress:  70%|████████████████▋       | 696/1000 [14:08<06:07,  1.21s/it]
Epoch Progress:  70%|████████████████▋       | 697/1000 [14:09<06:06,  1.21s/it]
Epoch Progress:  70%|████████████████▊       | 698/1000 [14:11<06:05,  1.21s/it]
Epoch Progress:  70%|████████████████▊       | 699/1000 [14:12<06:03,  1.21s/it]
Epoch Progress:  70%|████████████████▊       | 700/1000 [14:13<06:03,  1.21s/it]Dataset Size: 1%, Epoch: 701, Train Loss: 8.265819072723389, Val Loss: 11.01085899402569

Epoch Progress:  70%|████████████████▊       | 701/1000 [14:14<06:02,  1.21s/it]
Epoch Progress:  70%|████████████████▊       | 702/1000 [14:15<06:01,  1.21s/it]
Epoch Progress:  70%|████████████████▊       | 703/1000 [14:17<05:59,  1.21s/it]
Epoch Progress:  70%|████████████████▉       | 704/1000 [14:18<05:57,  1.21s/it]
Epoch Progress:  70%|████████████████▉       | 705/1000 [14:19<05:56,  1.21s/it]
Epoch Progress:  71%|████████████████▉       | 706/1000 [14:20<05:56,  1.21s/it]
Epoch Progress:  71%|████████████████▉       | 707/1000 [14:22<05:54,  1.21s/it]
Epoch Progress:  71%|████████████████▉       | 708/1000 [14:23<05:53,  1.21s/it]
Epoch Progress:  71%|█████████████████       | 709/1000 [14:24<05:51,  1.21s/it]
Epoch Progress:  71%|█████████████████       | 710/1000 [14:25<05:51,  1.21s/it]Dataset Size: 1%, Epoch: 711, Train Loss: 8.292383551597595, Val Loss: 11.006766628909421

Epoch Progress:  71%|█████████████████       | 711/1000 [14:26<05:50,  1.21s/it]
Epoch Progress:  71%|█████████████████       | 712/1000 [14:28<05:51,  1.22s/it]
Epoch Progress:  71%|█████████████████       | 713/1000 [14:29<05:49,  1.22s/it]
Epoch Progress:  71%|█████████████████▏      | 714/1000 [14:30<05:48,  1.22s/it]
Epoch Progress:  72%|█████████████████▏      | 715/1000 [14:31<05:46,  1.22s/it]
Epoch Progress:  72%|█████████████████▏      | 716/1000 [14:32<05:45,  1.22s/it]
Epoch Progress:  72%|█████████████████▏      | 717/1000 [14:34<05:43,  1.22s/it]
Epoch Progress:  72%|█████████████████▏      | 718/1000 [14:35<05:41,  1.21s/it]
Epoch Progress:  72%|█████████████████▎      | 719/1000 [14:36<05:40,  1.21s/it]
Epoch Progress:  72%|█████████████████▎      | 720/1000 [14:37<05:38,  1.21s/it]Dataset Size: 1%, Epoch: 721, Train Loss: 8.27016270160675, Val Loss: 11.008122976724204

Epoch Progress:  72%|█████████████████▎      | 721/1000 [14:38<05:36,  1.21s/it]
Epoch Progress:  72%|█████████████████▎      | 722/1000 [14:40<05:36,  1.21s/it]
Epoch Progress:  72%|█████████████████▎      | 723/1000 [14:41<05:35,  1.21s/it]
Epoch Progress:  72%|█████████████████▍      | 724/1000 [14:42<05:33,  1.21s/it]
Epoch Progress:  72%|█████████████████▍      | 725/1000 [14:43<05:32,  1.21s/it]
Epoch Progress:  73%|█████████████████▍      | 726/1000 [14:45<05:30,  1.21s/it]
Epoch Progress:  73%|█████████████████▍      | 727/1000 [14:46<05:29,  1.21s/it]
Epoch Progress:  73%|█████████████████▍      | 728/1000 [14:47<05:28,  1.21s/it]
Epoch Progress:  73%|█████████████████▍      | 729/1000 [14:48<05:27,  1.21s/it]
Epoch Progress:  73%|█████████████████▌      | 730/1000 [14:49<05:26,  1.21s/it]Dataset Size: 1%, Epoch: 731, Train Loss: 8.289439797401428, Val Loss: 11.009675954843496

Epoch Progress:  73%|█████████████████▌      | 731/1000 [14:51<05:27,  1.22s/it]
Epoch Progress:  73%|█████████████████▌      | 732/1000 [14:52<05:26,  1.22s/it]
Epoch Progress:  73%|█████████████████▌      | 733/1000 [14:53<05:24,  1.21s/it]
Epoch Progress:  73%|█████████████████▌      | 734/1000 [14:54<05:22,  1.21s/it]
Epoch Progress:  74%|█████████████████▋      | 735/1000 [14:55<05:20,  1.21s/it]
Epoch Progress:  74%|█████████████████▋      | 736/1000 [14:57<05:18,  1.21s/it]
Epoch Progress:  74%|█████████████████▋      | 737/1000 [14:58<05:18,  1.21s/it]
Epoch Progress:  74%|█████████████████▋      | 738/1000 [14:59<05:20,  1.22s/it]
Epoch Progress:  74%|█████████████████▋      | 739/1000 [15:00<05:20,  1.23s/it]
Epoch Progress:  74%|█████████████████▊      | 740/1000 [15:02<05:17,  1.22s/it]Dataset Size: 1%, Epoch: 741, Train Loss: 8.282012343406677, Val Loss: 11.017593383789062

Epoch Progress:  74%|█████████████████▊      | 741/1000 [15:03<05:15,  1.22s/it]
Epoch Progress:  74%|█████████████████▊      | 742/1000 [15:04<05:13,  1.22s/it]
Epoch Progress:  74%|█████████████████▊      | 743/1000 [15:05<05:11,  1.21s/it]
Epoch Progress:  74%|█████████████████▊      | 744/1000 [15:06<05:10,  1.21s/it]
Epoch Progress:  74%|█████████████████▉      | 745/1000 [15:08<05:08,  1.21s/it]
Epoch Progress:  75%|█████████████████▉      | 746/1000 [15:09<05:06,  1.21s/it]
Epoch Progress:  75%|█████████████████▉      | 747/1000 [15:10<05:05,  1.21s/it]
Epoch Progress:  75%|█████████████████▉      | 748/1000 [15:11<05:04,  1.21s/it]
Epoch Progress:  75%|█████████████████▉      | 749/1000 [15:12<05:03,  1.21s/it]
Epoch Progress:  75%|██████████████████      | 750/1000 [15:14<05:02,  1.21s/it]Dataset Size: 1%, Epoch: 751, Train Loss: 8.228878736495972, Val Loss: 11.015624962843857

Epoch Progress:  75%|██████████████████      | 751/1000 [15:15<05:01,  1.21s/it]
Epoch Progress:  75%|██████████████████      | 752/1000 [15:16<05:00,  1.21s/it]
Epoch Progress:  75%|██████████████████      | 753/1000 [15:17<04:58,  1.21s/it]
Epoch Progress:  75%|██████████████████      | 754/1000 [15:18<04:57,  1.21s/it]
Epoch Progress:  76%|██████████████████      | 755/1000 [15:20<04:57,  1.21s/it]
Epoch Progress:  76%|██████████████████▏     | 756/1000 [15:21<04:55,  1.21s/it]
Epoch Progress:  76%|██████████████████▏     | 757/1000 [15:22<04:53,  1.21s/it]
Epoch Progress:  76%|██████████████████▏     | 758/1000 [15:23<04:52,  1.21s/it]
Epoch Progress:  76%|██████████████████▏     | 759/1000 [15:25<04:52,  1.21s/it]
Epoch Progress:  76%|██████████████████▏     | 760/1000 [15:26<04:50,  1.21s/it]Dataset Size: 1%, Epoch: 761, Train Loss: 8.252196073532104, Val Loss: 11.020317350115095

Epoch Progress:  76%|██████████████████▎     | 761/1000 [15:27<04:49,  1.21s/it]
Epoch Progress:  76%|██████████████████▎     | 762/1000 [15:28<04:47,  1.21s/it]
Epoch Progress:  76%|██████████████████▎     | 763/1000 [15:29<04:46,  1.21s/it]
Epoch Progress:  76%|██████████████████▎     | 764/1000 [15:31<04:48,  1.22s/it]
Epoch Progress:  76%|██████████████████▎     | 765/1000 [15:32<04:46,  1.22s/it]
Epoch Progress:  77%|██████████████████▍     | 766/1000 [15:33<04:44,  1.22s/it]
Epoch Progress:  77%|██████████████████▍     | 767/1000 [15:34<04:42,  1.21s/it]
Epoch Progress:  77%|██████████████████▍     | 768/1000 [15:35<04:40,  1.21s/it]
Epoch Progress:  77%|██████████████████▍     | 769/1000 [15:37<04:39,  1.21s/it]
Epoch Progress:  77%|██████████████████▍     | 770/1000 [15:38<04:38,  1.21s/it]Dataset Size: 1%, Epoch: 771, Train Loss: 8.281563758850098, Val Loss: 11.018202769291864

Epoch Progress:  77%|██████████████████▌     | 771/1000 [15:39<04:36,  1.21s/it]
Epoch Progress:  77%|██████████████████▌     | 772/1000 [15:40<04:36,  1.21s/it]
Epoch Progress:  77%|██████████████████▌     | 773/1000 [15:41<04:34,  1.21s/it]
Epoch Progress:  77%|██████████████████▌     | 774/1000 [15:43<04:32,  1.21s/it]
Epoch Progress:  78%|██████████████████▌     | 775/1000 [15:44<04:31,  1.21s/it]
Epoch Progress:  78%|██████████████████▌     | 776/1000 [15:45<04:29,  1.20s/it]
Epoch Progress:  78%|██████████████████▋     | 777/1000 [15:46<04:28,  1.20s/it]
Epoch Progress:  78%|██████████████████▋     | 778/1000 [15:48<04:27,  1.20s/it]
Epoch Progress:  78%|██████████████████▋     | 779/1000 [15:49<04:25,  1.20s/it]
Epoch Progress:  78%|██████████████████▋     | 780/1000 [15:50<04:24,  1.20s/it]Dataset Size: 1%, Epoch: 781, Train Loss: 8.266835331916809, Val Loss: 11.020639568180233

Epoch Progress:  78%|██████████████████▋     | 781/1000 [15:51<04:23,  1.20s/it]
Epoch Progress:  78%|██████████████████▊     | 782/1000 [15:52<04:22,  1.21s/it]
Epoch Progress:  78%|██████████████████▊     | 783/1000 [15:54<04:21,  1.21s/it]
Epoch Progress:  78%|██████████████████▊     | 784/1000 [15:55<04:20,  1.20s/it]
Epoch Progress:  78%|██████████████████▊     | 785/1000 [15:56<04:18,  1.20s/it]
Epoch Progress:  79%|██████████████████▊     | 786/1000 [15:57<04:17,  1.20s/it]
Epoch Progress:  79%|██████████████████▉     | 787/1000 [15:58<04:16,  1.20s/it]
Epoch Progress:  79%|██████████████████▉     | 788/1000 [16:00<04:15,  1.21s/it]
Epoch Progress:  79%|██████████████████▉     | 789/1000 [16:01<04:14,  1.20s/it]
Epoch Progress:  79%|██████████████████▉     | 790/1000 [16:02<04:14,  1.21s/it]Dataset Size: 1%, Epoch: 791, Train Loss: 8.251153469085693, Val Loss: 11.018346018605419

Epoch Progress:  79%|██████████████████▉     | 791/1000 [16:03<04:13,  1.21s/it]
Epoch Progress:  79%|███████████████████     | 792/1000 [16:04<04:11,  1.21s/it]
Epoch Progress:  79%|███████████████████     | 793/1000 [16:06<04:10,  1.21s/it]
Epoch Progress:  79%|███████████████████     | 794/1000 [16:07<04:08,  1.21s/it]
Epoch Progress:  80%|███████████████████     | 795/1000 [16:08<04:07,  1.21s/it]
Epoch Progress:  80%|███████████████████     | 796/1000 [16:09<04:05,  1.21s/it]
Epoch Progress:  80%|███████████████████▏    | 797/1000 [16:10<04:06,  1.21s/it]
Epoch Progress:  80%|███████████████████▏    | 798/1000 [16:12<04:04,  1.21s/it]
Epoch Progress:  80%|███████████████████▏    | 799/1000 [16:13<04:03,  1.21s/it]
Epoch Progress:  80%|███████████████████▏    | 800/1000 [16:14<04:02,  1.21s/it]Dataset Size: 1%, Epoch: 801, Train Loss: 8.267253518104553, Val Loss: 11.02109803781881

Epoch Progress:  80%|███████████████████▏    | 801/1000 [16:15<04:00,  1.21s/it]
Epoch Progress:  80%|███████████████████▏    | 802/1000 [16:16<03:59,  1.21s/it]
Epoch Progress:  80%|███████████████████▎    | 803/1000 [16:18<03:57,  1.21s/it]
Epoch Progress:  80%|███████████████████▎    | 804/1000 [16:19<03:56,  1.21s/it]
Epoch Progress:  80%|███████████████████▎    | 805/1000 [16:20<03:56,  1.21s/it]
Epoch Progress:  81%|███████████████████▎    | 806/1000 [16:21<03:55,  1.21s/it]
Epoch Progress:  81%|███████████████████▎    | 807/1000 [16:23<03:53,  1.21s/it]
Epoch Progress:  81%|███████████████████▍    | 808/1000 [16:24<03:52,  1.21s/it]
Epoch Progress:  81%|███████████████████▍    | 809/1000 [16:25<03:51,  1.21s/it]
Epoch Progress:  81%|███████████████████▍    | 810/1000 [16:26<03:49,  1.21s/it]Dataset Size: 1%, Epoch: 811, Train Loss: 8.259608387947083, Val Loss: 11.0241688938884

Epoch Progress:  81%|███████████████████▍    | 811/1000 [16:27<03:47,  1.20s/it]
Epoch Progress:  81%|███████████████████▍    | 812/1000 [16:29<03:46,  1.20s/it]
Epoch Progress:  81%|███████████████████▌    | 813/1000 [16:30<03:45,  1.21s/it]
Epoch Progress:  81%|███████████████████▌    | 814/1000 [16:31<03:44,  1.21s/it]
Epoch Progress:  82%|███████████████████▌    | 815/1000 [16:32<03:42,  1.20s/it]
Epoch Progress:  82%|███████████████████▌    | 816/1000 [16:33<03:42,  1.21s/it]
Epoch Progress:  82%|███████████████████▌    | 817/1000 [16:35<03:41,  1.21s/it]
Epoch Progress:  82%|███████████████████▋    | 818/1000 [16:36<03:41,  1.21s/it]
Epoch Progress:  82%|███████████████████▋    | 819/1000 [16:37<03:39,  1.21s/it]
Epoch Progress:  82%|███████████████████▋    | 820/1000 [16:38<03:38,  1.21s/it]Dataset Size: 1%, Epoch: 821, Train Loss: 8.252972722053528, Val Loss: 11.027651390472016

Epoch Progress:  82%|███████████████████▋    | 821/1000 [16:39<03:37,  1.21s/it]
Epoch Progress:  82%|███████████████████▋    | 822/1000 [16:41<03:35,  1.21s/it]
Epoch Progress:  82%|███████████████████▊    | 823/1000 [16:42<03:34,  1.21s/it]
Epoch Progress:  82%|███████████████████▊    | 824/1000 [16:43<03:33,  1.21s/it]
Epoch Progress:  82%|███████████████████▊    | 825/1000 [16:44<03:31,  1.21s/it]
Epoch Progress:  83%|███████████████████▊    | 826/1000 [16:46<03:30,  1.21s/it]
Epoch Progress:  83%|███████████████████▊    | 827/1000 [16:47<03:28,  1.21s/it]
Epoch Progress:  83%|███████████████████▊    | 828/1000 [16:48<03:26,  1.20s/it]
Epoch Progress:  83%|███████████████████▉    | 829/1000 [16:49<03:25,  1.20s/it]
Epoch Progress:  83%|███████████████████▉    | 830/1000 [16:50<03:25,  1.21s/it]Dataset Size: 1%, Epoch: 831, Train Loss: 8.301655173301697, Val Loss: 11.026955431157893

Epoch Progress:  83%|███████████████████▉    | 831/1000 [16:52<03:24,  1.21s/it]
Epoch Progress:  83%|███████████████████▉    | 832/1000 [16:53<03:22,  1.21s/it]
Epoch Progress:  83%|███████████████████▉    | 833/1000 [16:54<03:22,  1.21s/it]
Epoch Progress:  83%|████████████████████    | 834/1000 [16:55<03:20,  1.21s/it]
Epoch Progress:  84%|████████████████████    | 835/1000 [16:56<03:19,  1.21s/it]
Epoch Progress:  84%|████████████████████    | 836/1000 [16:58<03:17,  1.21s/it]
Epoch Progress:  84%|████████████████████    | 837/1000 [16:59<03:16,  1.20s/it]
Epoch Progress:  84%|████████████████████    | 838/1000 [17:00<03:15,  1.21s/it]
Epoch Progress:  84%|████████████████████▏   | 839/1000 [17:01<03:14,  1.21s/it]
Epoch Progress:  84%|████████████████████▏   | 840/1000 [17:02<03:13,  1.21s/it]Dataset Size: 1%, Epoch: 841, Train Loss: 8.27822756767273, Val Loss: 11.025219260872184

Epoch Progress:  84%|████████████████████▏   | 841/1000 [17:04<03:11,  1.21s/it]
Epoch Progress:  84%|████████████████████▏   | 842/1000 [17:05<03:10,  1.21s/it]
Epoch Progress:  84%|████████████████████▏   | 843/1000 [17:06<03:10,  1.21s/it]
Epoch Progress:  84%|████████████████████▎   | 844/1000 [17:07<03:08,  1.21s/it]
Epoch Progress:  84%|████████████████████▎   | 845/1000 [17:08<03:07,  1.21s/it]
Epoch Progress:  85%|████████████████████▎   | 846/1000 [17:10<03:07,  1.21s/it]
Epoch Progress:  85%|████████████████████▎   | 847/1000 [17:11<03:05,  1.21s/it]
Epoch Progress:  85%|████████████████████▎   | 848/1000 [17:12<03:04,  1.21s/it]
Epoch Progress:  85%|████████████████████▍   | 849/1000 [17:13<03:03,  1.21s/it]
Epoch Progress:  85%|████████████████████▍   | 850/1000 [17:15<03:01,  1.21s/it]Dataset Size: 1%, Epoch: 851, Train Loss: 8.259573578834534, Val Loss: 11.026896179496468

Epoch Progress:  85%|████████████████████▍   | 851/1000 [17:16<02:59,  1.21s/it]
Epoch Progress:  85%|████████████████████▍   | 852/1000 [17:17<02:58,  1.21s/it]
Epoch Progress:  85%|████████████████████▍   | 853/1000 [17:18<02:57,  1.20s/it]
Epoch Progress:  85%|████████████████████▍   | 854/1000 [17:19<02:55,  1.21s/it]
Epoch Progress:  86%|████████████████████▌   | 855/1000 [17:21<02:54,  1.21s/it]
Epoch Progress:  86%|████████████████████▌   | 856/1000 [17:22<02:53,  1.20s/it]
Epoch Progress:  86%|████████████████████▌   | 857/1000 [17:23<02:52,  1.21s/it]
Epoch Progress:  86%|████████████████████▌   | 858/1000 [17:24<02:51,  1.21s/it]
Epoch Progress:  86%|████████████████████▌   | 859/1000 [17:25<02:50,  1.21s/it]
Epoch Progress:  86%|████████████████████▋   | 860/1000 [17:27<02:48,  1.20s/it]Dataset Size: 1%, Epoch: 861, Train Loss: 8.220863461494446, Val Loss: 11.025241641255167

Epoch Progress:  86%|████████████████████▋   | 861/1000 [17:28<02:46,  1.20s/it]
Epoch Progress:  86%|████████████████████▋   | 862/1000 [17:29<02:45,  1.20s/it]
Epoch Progress:  86%|████████████████████▋   | 863/1000 [17:30<02:44,  1.20s/it]
Epoch Progress:  86%|████████████████████▋   | 864/1000 [17:31<02:43,  1.20s/it]
Epoch Progress:  86%|████████████████████▊   | 865/1000 [17:33<02:42,  1.20s/it]
Epoch Progress:  87%|████████████████████▊   | 866/1000 [17:34<02:40,  1.20s/it]
Epoch Progress:  87%|████████████████████▊   | 867/1000 [17:35<02:39,  1.20s/it]
Epoch Progress:  87%|████████████████████▊   | 868/1000 [17:36<02:38,  1.20s/it]
Epoch Progress:  87%|████████████████████▊   | 869/1000 [17:37<02:38,  1.21s/it]
Epoch Progress:  87%|████████████████████▉   | 870/1000 [17:39<02:37,  1.21s/it]Dataset Size: 1%, Epoch: 871, Train Loss: 8.265900254249573, Val Loss: 11.02953795643596

Epoch Progress:  87%|████████████████████▉   | 871/1000 [17:40<02:36,  1.21s/it]
Epoch Progress:  87%|████████████████████▉   | 872/1000 [17:41<02:35,  1.21s/it]
Epoch Progress:  87%|████████████████████▉   | 873/1000 [17:42<02:34,  1.21s/it]
Epoch Progress:  87%|████████████████████▉   | 874/1000 [17:43<02:32,  1.21s/it]
Epoch Progress:  88%|█████████████████████   | 875/1000 [17:45<02:31,  1.21s/it]
Epoch Progress:  88%|█████████████████████   | 876/1000 [17:46<02:29,  1.21s/it]
Epoch Progress:  88%|█████████████████████   | 877/1000 [17:47<02:28,  1.21s/it]
Epoch Progress:  88%|█████████████████████   | 878/1000 [17:48<02:27,  1.21s/it]
Epoch Progress:  88%|█████████████████████   | 879/1000 [17:50<02:26,  1.21s/it]
Epoch Progress:  88%|█████████████████████   | 880/1000 [17:51<02:25,  1.21s/it]Dataset Size: 1%, Epoch: 881, Train Loss: 8.266232371330261, Val Loss: 11.032930807633834

Epoch Progress:  88%|█████████████████████▏  | 881/1000 [17:52<02:24,  1.21s/it]
Epoch Progress:  88%|█████████████████████▏  | 882/1000 [17:53<02:22,  1.21s/it]
Epoch Progress:  88%|█████████████████████▏  | 883/1000 [17:54<02:20,  1.20s/it]
Epoch Progress:  88%|█████████████████████▏  | 884/1000 [17:56<02:19,  1.21s/it]
Epoch Progress:  88%|█████████████████████▏  | 885/1000 [17:57<02:18,  1.20s/it]
Epoch Progress:  89%|█████████████████████▎  | 886/1000 [17:58<02:17,  1.20s/it]
Epoch Progress:  89%|█████████████████████▎  | 887/1000 [17:59<02:16,  1.20s/it]
Epoch Progress:  89%|█████████████████████▎  | 888/1000 [18:00<02:15,  1.21s/it]
Epoch Progress:  89%|█████████████████████▎  | 889/1000 [18:02<02:13,  1.20s/it]
Epoch Progress:  89%|█████████████████████▎  | 890/1000 [18:03<02:12,  1.20s/it]Dataset Size: 1%, Epoch: 891, Train Loss: 8.248407363891602, Val Loss: 11.028784194549957

Epoch Progress:  89%|█████████████████████▍  | 891/1000 [18:04<02:11,  1.20s/it]
Epoch Progress:  89%|█████████████████████▍  | 892/1000 [18:05<02:09,  1.20s/it]
Epoch Progress:  89%|█████████████████████▍  | 893/1000 [18:06<02:08,  1.20s/it]
Epoch Progress:  89%|█████████████████████▍  | 894/1000 [18:08<02:07,  1.20s/it]
Epoch Progress:  90%|█████████████████████▍  | 895/1000 [18:09<02:06,  1.21s/it]
Epoch Progress:  90%|█████████████████████▌  | 896/1000 [18:10<02:05,  1.21s/it]
Epoch Progress:  90%|█████████████████████▌  | 897/1000 [18:11<02:04,  1.21s/it]
Epoch Progress:  90%|█████████████████████▌  | 898/1000 [18:12<02:03,  1.21s/it]
Epoch Progress:  90%|█████████████████████▌  | 899/1000 [18:14<02:01,  1.21s/it]
Epoch Progress:  90%|█████████████████████▌  | 900/1000 [18:15<02:01,  1.21s/it]Dataset Size: 1%, Epoch: 901, Train Loss: 8.249261677265167, Val Loss: 11.030043924009645

Epoch Progress:  90%|█████████████████████▌  | 901/1000 [18:16<01:59,  1.21s/it]
Epoch Progress:  90%|█████████████████████▋  | 902/1000 [18:17<01:58,  1.21s/it]
Epoch Progress:  90%|█████████████████████▋  | 903/1000 [18:18<01:57,  1.21s/it]
Epoch Progress:  90%|█████████████████████▋  | 904/1000 [18:20<01:56,  1.21s/it]
Epoch Progress:  90%|█████████████████████▋  | 905/1000 [18:21<01:55,  1.21s/it]
Epoch Progress:  91%|█████████████████████▋  | 906/1000 [18:22<01:53,  1.21s/it]
Epoch Progress:  91%|█████████████████████▊  | 907/1000 [18:23<01:52,  1.21s/it]
Epoch Progress:  91%|█████████████████████▊  | 908/1000 [18:25<01:51,  1.21s/it]
Epoch Progress:  91%|█████████████████████▊  | 909/1000 [18:26<01:50,  1.22s/it]
Epoch Progress:  91%|█████████████████████▊  | 910/1000 [18:27<01:49,  1.21s/it]Dataset Size: 1%, Epoch: 911, Train Loss: 8.260085105895996, Val Loss: 11.030814344232732

Epoch Progress:  91%|█████████████████████▊  | 911/1000 [18:28<01:47,  1.21s/it]
Epoch Progress:  91%|█████████████████████▉  | 912/1000 [18:29<01:46,  1.21s/it]
Epoch Progress:  91%|█████████████████████▉  | 913/1000 [18:31<01:45,  1.21s/it]
Epoch Progress:  91%|█████████████████████▉  | 914/1000 [18:32<01:44,  1.21s/it]
Epoch Progress:  92%|█████████████████████▉  | 915/1000 [18:33<01:42,  1.21s/it]
Epoch Progress:  92%|█████████████████████▉  | 916/1000 [18:34<01:41,  1.21s/it]
Epoch Progress:  92%|██████████████████████  | 917/1000 [18:35<01:40,  1.21s/it]
Epoch Progress:  92%|██████████████████████  | 918/1000 [18:37<01:39,  1.21s/it]
Epoch Progress:  92%|██████████████████████  | 919/1000 [18:38<01:37,  1.21s/it]
Epoch Progress:  92%|██████████████████████  | 920/1000 [18:39<01:36,  1.21s/it]Dataset Size: 1%, Epoch: 921, Train Loss: 8.239554762840271, Val Loss: 11.025722986691958

Epoch Progress:  92%|██████████████████████  | 921/1000 [18:40<01:36,  1.22s/it]
Epoch Progress:  92%|██████████████████████▏ | 922/1000 [18:42<01:34,  1.22s/it]
Epoch Progress:  92%|██████████████████████▏ | 923/1000 [18:43<01:33,  1.21s/it]
Epoch Progress:  92%|██████████████████████▏ | 924/1000 [18:44<01:32,  1.21s/it]
Epoch Progress:  92%|██████████████████████▏ | 925/1000 [18:45<01:30,  1.21s/it]
Epoch Progress:  93%|██████████████████████▏ | 926/1000 [18:46<01:29,  1.20s/it]
Epoch Progress:  93%|██████████████████████▏ | 927/1000 [18:48<01:28,  1.21s/it]
Epoch Progress:  93%|██████████████████████▎ | 928/1000 [18:49<01:27,  1.21s/it]
Epoch Progress:  93%|██████████████████████▎ | 929/1000 [18:50<01:26,  1.21s/it]
Epoch Progress:  93%|██████████████████████▎ | 930/1000 [18:51<01:24,  1.21s/it]Dataset Size: 1%, Epoch: 931, Train Loss: 8.24958622455597, Val Loss: 11.029916218348912

Epoch Progress:  93%|██████████████████████▎ | 931/1000 [18:52<01:23,  1.21s/it]
Epoch Progress:  93%|██████████████████████▎ | 932/1000 [18:54<01:22,  1.21s/it]
Epoch Progress:  93%|██████████████████████▍ | 933/1000 [18:55<01:21,  1.21s/it]
Epoch Progress:  93%|██████████████████████▍ | 934/1000 [18:56<01:19,  1.21s/it]
Epoch Progress:  94%|██████████████████████▍ | 935/1000 [18:57<01:18,  1.21s/it]
Epoch Progress:  94%|██████████████████████▍ | 936/1000 [18:58<01:17,  1.21s/it]
Epoch Progress:  94%|██████████████████████▍ | 937/1000 [19:00<01:16,  1.21s/it]
Epoch Progress:  94%|██████████████████████▌ | 938/1000 [19:01<01:15,  1.21s/it]
Epoch Progress:  94%|██████████████████████▌ | 939/1000 [19:02<01:13,  1.21s/it]
Epoch Progress:  94%|██████████████████████▌ | 940/1000 [19:03<01:12,  1.21s/it]Dataset Size: 1%, Epoch: 941, Train Loss: 8.228804588317871, Val Loss: 11.023391797945097

Epoch Progress:  94%|██████████████████████▌ | 941/1000 [19:05<01:11,  1.21s/it]
Epoch Progress:  94%|██████████████████████▌ | 942/1000 [19:06<01:10,  1.21s/it]
Epoch Progress:  94%|██████████████████████▋ | 943/1000 [19:07<01:08,  1.21s/it]
Epoch Progress:  94%|██████████████████████▋ | 944/1000 [19:08<01:07,  1.21s/it]
Epoch Progress:  94%|██████████████████████▋ | 945/1000 [19:09<01:06,  1.20s/it]
Epoch Progress:  95%|██████████████████████▋ | 946/1000 [19:11<01:05,  1.21s/it]
Epoch Progress:  95%|██████████████████████▋ | 947/1000 [19:12<01:04,  1.22s/it]
Epoch Progress:  95%|██████████████████████▊ | 948/1000 [19:13<01:03,  1.22s/it]
Epoch Progress:  95%|██████████████████████▊ | 949/1000 [19:14<01:02,  1.22s/it]
Epoch Progress:  95%|██████████████████████▊ | 950/1000 [19:15<01:00,  1.21s/it]Dataset Size: 1%, Epoch: 951, Train Loss: 8.244904637336731, Val Loss: 11.031378015295251

Epoch Progress:  95%|██████████████████████▊ | 951/1000 [19:17<00:59,  1.21s/it]
Epoch Progress:  95%|██████████████████████▊ | 952/1000 [19:18<00:58,  1.22s/it]
Epoch Progress:  95%|██████████████████████▊ | 953/1000 [19:19<00:57,  1.21s/it]
Epoch Progress:  95%|██████████████████████▉ | 954/1000 [19:20<00:56,  1.22s/it]
Epoch Progress:  96%|██████████████████████▉ | 955/1000 [19:22<00:54,  1.22s/it]
Epoch Progress:  96%|██████████████████████▉ | 956/1000 [19:23<00:53,  1.22s/it]
Epoch Progress:  96%|██████████████████████▉ | 957/1000 [19:24<00:52,  1.22s/it]
Epoch Progress:  96%|██████████████████████▉ | 958/1000 [19:25<00:51,  1.22s/it]
Epoch Progress:  96%|███████████████████████ | 959/1000 [19:26<00:49,  1.21s/it]
Epoch Progress:  96%|███████████████████████ | 960/1000 [19:28<00:48,  1.21s/it]Dataset Size: 1%, Epoch: 961, Train Loss: 8.25989055633545, Val Loss: 11.027633642221426

Epoch Progress:  96%|███████████████████████ | 961/1000 [19:29<00:47,  1.21s/it]
Epoch Progress:  96%|███████████████████████ | 962/1000 [19:30<00:46,  1.21s/it]
Epoch Progress:  96%|███████████████████████ | 963/1000 [19:31<00:44,  1.21s/it]
Epoch Progress:  96%|███████████████████████▏| 964/1000 [19:32<00:43,  1.21s/it]
Epoch Progress:  96%|███████████████████████▏| 965/1000 [19:34<00:42,  1.21s/it]
Epoch Progress:  97%|███████████████████████▏| 966/1000 [19:35<00:41,  1.21s/it]
Epoch Progress:  97%|███████████████████████▏| 967/1000 [19:36<00:40,  1.22s/it]
Epoch Progress:  97%|███████████████████████▏| 968/1000 [19:37<00:38,  1.22s/it]
Epoch Progress:  97%|███████████████████████▎| 969/1000 [19:39<00:37,  1.22s/it]
Epoch Progress:  97%|███████████████████████▎| 970/1000 [19:40<00:36,  1.22s/it]Dataset Size: 1%, Epoch: 971, Train Loss: 8.26240074634552, Val Loss: 11.028475129759157

Epoch Progress:  97%|███████████████████████▎| 971/1000 [19:41<00:35,  1.22s/it]
Epoch Progress:  97%|███████████████████████▎| 972/1000 [19:42<00:34,  1.22s/it]
Epoch Progress:  97%|███████████████████████▎| 973/1000 [19:43<00:33,  1.22s/it]
Epoch Progress:  97%|███████████████████████▍| 974/1000 [19:45<00:31,  1.22s/it]
Epoch Progress:  98%|███████████████████████▍| 975/1000 [19:46<00:30,  1.22s/it]
Epoch Progress:  98%|███████████████████████▍| 976/1000 [19:47<00:29,  1.22s/it]
Epoch Progress:  98%|███████████████████████▍| 977/1000 [19:48<00:28,  1.22s/it]
Epoch Progress:  98%|███████████████████████▍| 978/1000 [19:50<00:26,  1.22s/it]
Epoch Progress:  98%|███████████████████████▍| 979/1000 [19:51<00:25,  1.22s/it]
Epoch Progress:  98%|███████████████████████▌| 980/1000 [19:52<00:24,  1.22s/it]Dataset Size: 1%, Epoch: 981, Train Loss: 8.25956392288208, Val Loss: 11.025439844503031

Epoch Progress:  98%|███████████████████████▌| 981/1000 [19:53<00:23,  1.22s/it]
Epoch Progress:  98%|███████████████████████▌| 982/1000 [19:54<00:22,  1.22s/it]
Epoch Progress:  98%|███████████████████████▌| 983/1000 [19:56<00:20,  1.22s/it]
Epoch Progress:  98%|███████████████████████▌| 984/1000 [19:57<00:19,  1.22s/it]
Epoch Progress:  98%|███████████████████████▋| 985/1000 [19:58<00:18,  1.21s/it]
Epoch Progress:  99%|███████████████████████▋| 986/1000 [19:59<00:17,  1.22s/it]
Epoch Progress:  99%|███████████████████████▋| 987/1000 [20:00<00:15,  1.22s/it]
Epoch Progress:  99%|███████████████████████▋| 988/1000 [20:02<00:14,  1.22s/it]
Epoch Progress:  99%|███████████████████████▋| 989/1000 [20:03<00:13,  1.22s/it]
Epoch Progress:  99%|███████████████████████▊| 990/1000 [20:04<00:12,  1.22s/it]Dataset Size: 1%, Epoch: 991, Train Loss: 8.260830044746399, Val Loss: 11.026842687037083

Epoch Progress:  99%|███████████████████████▊| 991/1000 [20:05<00:10,  1.22s/it]
Epoch Progress:  99%|███████████████████████▊| 992/1000 [20:07<00:09,  1.21s/it]
Epoch Progress:  99%|███████████████████████▊| 993/1000 [20:08<00:08,  1.21s/it]
Epoch Progress:  99%|███████████████████████▊| 994/1000 [20:09<00:07,  1.21s/it]
Epoch Progress: 100%|███████████████████████▉| 995/1000 [20:10<00:06,  1.22s/it]
Epoch Progress: 100%|███████████████████████▉| 996/1000 [20:11<00:04,  1.22s/it]
Epoch Progress: 100%|███████████████████████▉| 997/1000 [20:13<00:03,  1.22s/it]
Epoch Progress: 100%|███████████████████████▉| 998/1000 [20:14<00:02,  1.22s/it]
Epoch Progress: 100%|███████████████████████▉| 999/1000 [20:15<00:01,  1.23s/it]
Epoch Progress: 100%|███████████████████████| 1000/1000 [20:16<00:00,  1.22s/it]
Dataset Size: 1%, Val Perplexity: 41726.71875
"""

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
plt.title('Params=200k, Fraction=0.01, LR=0.001')
plt.legend()
plt.grid(True)
plt.savefig('200k_small_f_0.01_lr_0.001.png')
plt.show()
