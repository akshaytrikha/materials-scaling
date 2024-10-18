import re
import pandas as pd

log_data = '''
Data Iteration:   0%|                                     | 0/1 [00:00<?, ?it/s]
Model is on device cuda and has 1622688 parameters

Epoch Progress:   0%|                                  | 0/1000 [00:00<?, ?it/s]Dataset Size: 10%, Epoch: 1, Train Loss: 11.008990112103914, Val Loss: 11.003721334995367

Epoch Progress:   0%|                          | 1/1000 [00:03<52:34,  3.16s/it]
Epoch Progress:   0%|                          | 2/1000 [00:05<48:49,  2.93s/it]
Epoch Progress:   0%|                          | 3/1000 [00:08<47:37,  2.87s/it]
Epoch Progress:   0%|                          | 4/1000 [00:11<47:18,  2.85s/it]
Epoch Progress:   0%|▏                         | 5/1000 [00:14<46:46,  2.82s/it]
Epoch Progress:   1%|▏                         | 6/1000 [00:17<46:32,  2.81s/it]
Epoch Progress:   1%|▏                         | 7/1000 [00:19<46:17,  2.80s/it]
Epoch Progress:   1%|▏                         | 8/1000 [00:22<46:08,  2.79s/it]
Epoch Progress:   1%|▏                         | 9/1000 [00:25<46:01,  2.79s/it]
Epoch Progress:   1%|▎                        | 10/1000 [00:28<46:06,  2.79s/it]Dataset Size: 10%, Epoch: 11, Train Loss: 10.64789925123516, Val Loss: 10.611736713311611

Epoch Progress:   1%|▎                        | 11/1000 [00:31<45:57,  2.79s/it]
Epoch Progress:   1%|▎                        | 12/1000 [00:33<45:56,  2.79s/it]
Epoch Progress:   1%|▎                        | 13/1000 [00:36<45:46,  2.78s/it]
Epoch Progress:   1%|▎                        | 14/1000 [00:39<45:38,  2.78s/it]
Epoch Progress:   2%|▍                        | 15/1000 [00:42<45:35,  2.78s/it]
Epoch Progress:   2%|▍                        | 16/1000 [00:44<45:42,  2.79s/it]
Epoch Progress:   2%|▍                        | 17/1000 [00:47<45:38,  2.79s/it]
Epoch Progress:   2%|▍                        | 18/1000 [00:50<45:41,  2.79s/it]
Epoch Progress:   2%|▍                        | 19/1000 [00:53<45:34,  2.79s/it]
Epoch Progress:   2%|▌                        | 20/1000 [00:56<45:36,  2.79s/it]Dataset Size: 10%, Epoch: 21, Train Loss: 8.900559801804391, Val Loss: 8.999526072771122

Epoch Progress:   2%|▌                        | 21/1000 [00:58<45:28,  2.79s/it]
Epoch Progress:   2%|▌                        | 22/1000 [01:01<45:24,  2.79s/it]
Epoch Progress:   2%|▌                        | 23/1000 [01:04<45:21,  2.79s/it]
Epoch Progress:   2%|▌                        | 24/1000 [01:07<45:25,  2.79s/it]
Epoch Progress:   2%|▋                        | 25/1000 [01:10<45:16,  2.79s/it]
Epoch Progress:   3%|▋                        | 26/1000 [01:12<45:11,  2.78s/it]
Epoch Progress:   3%|▋                        | 27/1000 [01:15<45:07,  2.78s/it]
Epoch Progress:   3%|▋                        | 28/1000 [01:18<45:13,  2.79s/it]
Epoch Progress:   3%|▋                        | 29/1000 [01:21<45:12,  2.79s/it]
Epoch Progress:   3%|▊                        | 30/1000 [01:23<45:10,  2.79s/it]Dataset Size: 10%, Epoch: 31, Train Loss: 7.474001206849751, Val Loss: 7.7779233027727175

Epoch Progress:   3%|▊                        | 31/1000 [01:26<45:05,  2.79s/it]
Epoch Progress:   3%|▊                        | 32/1000 [01:29<45:03,  2.79s/it]
Epoch Progress:   3%|▊                        | 33/1000 [01:32<44:59,  2.79s/it]
Epoch Progress:   3%|▊                        | 34/1000 [01:35<44:56,  2.79s/it]
Epoch Progress:   4%|▉                        | 35/1000 [01:37<44:46,  2.78s/it]
Epoch Progress:   4%|▉                        | 36/1000 [01:40<44:39,  2.78s/it]
Epoch Progress:   4%|▉                        | 37/1000 [01:43<44:35,  2.78s/it]
Epoch Progress:   4%|▉                        | 38/1000 [01:46<44:33,  2.78s/it]
Epoch Progress:   4%|▉                        | 39/1000 [01:49<44:30,  2.78s/it]
Epoch Progress:   4%|█                        | 40/1000 [01:51<44:35,  2.79s/it]Dataset Size: 10%, Epoch: 41, Train Loss: 7.0563266026346305, Val Loss: 7.5203315416971845

Epoch Progress:   4%|█                        | 41/1000 [01:54<44:30,  2.78s/it]
Epoch Progress:   4%|█                        | 42/1000 [01:57<44:36,  2.79s/it]
Epoch Progress:   4%|█                        | 43/1000 [02:00<44:28,  2.79s/it]
Epoch Progress:   4%|█                        | 44/1000 [02:03<44:31,  2.79s/it]
Epoch Progress:   4%|█▏                       | 45/1000 [02:05<44:20,  2.79s/it]
Epoch Progress:   5%|█▏                       | 46/1000 [02:08<44:20,  2.79s/it]
Epoch Progress:   5%|█▏                       | 47/1000 [02:11<44:15,  2.79s/it]
Epoch Progress:   5%|█▏                       | 48/1000 [02:14<44:15,  2.79s/it]
Epoch Progress:   5%|█▏                       | 49/1000 [02:16<44:07,  2.78s/it]
Epoch Progress:   5%|█▎                       | 50/1000 [02:19<44:03,  2.78s/it]Dataset Size: 10%, Epoch: 51, Train Loss: 6.686705576746087, Val Loss: 7.322966673435309

Epoch Progress:   5%|█▎                       | 51/1000 [02:22<44:00,  2.78s/it]
Epoch Progress:   5%|█▎                       | 52/1000 [02:25<44:12,  2.80s/it]
Epoch Progress:   5%|█▎                       | 53/1000 [02:28<44:03,  2.79s/it]
Epoch Progress:   5%|█▎                       | 54/1000 [02:30<43:52,  2.78s/it]
Epoch Progress:   6%|█▍                       | 55/1000 [02:33<43:50,  2.78s/it]
Epoch Progress:   6%|█▍                       | 56/1000 [02:36<43:55,  2.79s/it]
Epoch Progress:   6%|█▍                       | 57/1000 [02:39<43:48,  2.79s/it]
Epoch Progress:   6%|█▍                       | 58/1000 [02:42<43:51,  2.79s/it]
Epoch Progress:   6%|█▍                       | 59/1000 [02:44<43:44,  2.79s/it]
Epoch Progress:   6%|█▌                       | 60/1000 [02:47<43:40,  2.79s/it]Dataset Size: 10%, Epoch: 61, Train Loss: 6.2621661236411645, Val Loss: 7.195631601871589

Epoch Progress:   6%|█▌                       | 61/1000 [02:50<43:37,  2.79s/it]
Epoch Progress:   6%|█▌                       | 62/1000 [02:53<43:36,  2.79s/it]
Epoch Progress:   6%|█▌                       | 63/1000 [02:55<43:36,  2.79s/it]
Epoch Progress:   6%|█▌                       | 64/1000 [02:58<43:42,  2.80s/it]
Epoch Progress:   6%|█▋                       | 65/1000 [03:01<43:34,  2.80s/it]
Epoch Progress:   7%|█▋                       | 66/1000 [03:04<43:33,  2.80s/it]
Epoch Progress:   7%|█▋                       | 67/1000 [03:07<43:22,  2.79s/it]
Epoch Progress:   7%|█▋                       | 68/1000 [03:09<43:23,  2.79s/it]
Epoch Progress:   7%|█▋                       | 69/1000 [03:12<43:04,  2.78s/it]
Epoch Progress:   7%|█▊                       | 70/1000 [03:15<42:48,  2.76s/it]Dataset Size: 10%, Epoch: 71, Train Loss: 5.846643711391248, Val Loss: 7.174742992107685

Epoch Progress:   7%|█▊                       | 71/1000 [03:18<42:36,  2.75s/it]
Epoch Progress:   7%|█▊                       | 72/1000 [03:20<42:28,  2.75s/it]
Epoch Progress:   7%|█▊                       | 73/1000 [03:23<42:20,  2.74s/it]
Epoch Progress:   7%|█▊                       | 74/1000 [03:26<42:18,  2.74s/it]
Epoch Progress:   8%|█▉                       | 75/1000 [03:29<42:10,  2.74s/it]
Epoch Progress:   8%|█▉                       | 76/1000 [03:31<42:09,  2.74s/it]
Epoch Progress:   8%|█▉                       | 77/1000 [03:34<42:03,  2.73s/it]
Epoch Progress:   8%|█▉                       | 78/1000 [03:37<41:58,  2.73s/it]
Epoch Progress:   8%|█▉                       | 79/1000 [03:39<41:53,  2.73s/it]
Epoch Progress:   8%|██                       | 80/1000 [03:42<41:52,  2.73s/it]Dataset Size: 10%, Epoch: 81, Train Loss: 5.472928398533871, Val Loss: 7.234599749247233

Epoch Progress:   8%|██                       | 81/1000 [03:45<41:47,  2.73s/it]
Epoch Progress:   8%|██                       | 82/1000 [03:48<41:43,  2.73s/it]
Epoch Progress:   8%|██                       | 83/1000 [03:50<41:39,  2.73s/it]
Epoch Progress:   8%|██                       | 84/1000 [03:53<41:39,  2.73s/it]
Epoch Progress:   8%|██▏                      | 85/1000 [03:56<41:36,  2.73s/it]
Epoch Progress:   9%|██▏                      | 86/1000 [03:59<41:37,  2.73s/it]
Epoch Progress:   9%|██▏                      | 87/1000 [04:01<41:36,  2.73s/it]
Epoch Progress:   9%|██▏                      | 88/1000 [04:04<41:32,  2.73s/it]
Epoch Progress:   9%|██▏                      | 89/1000 [04:07<41:28,  2.73s/it]
Epoch Progress:   9%|██▎                      | 90/1000 [04:10<41:31,  2.74s/it]Dataset Size: 10%, Epoch: 91, Train Loss: 5.154121248345626, Val Loss: 7.366555311740973

Epoch Progress:   9%|██▎                      | 91/1000 [04:12<41:30,  2.74s/it]
Epoch Progress:   9%|██▎                      | 92/1000 [04:15<41:30,  2.74s/it]
Epoch Progress:   9%|██▎                      | 93/1000 [04:18<41:29,  2.74s/it]
Epoch Progress:   9%|██▎                      | 94/1000 [04:21<41:27,  2.75s/it]
Epoch Progress:  10%|██▍                      | 95/1000 [04:23<41:23,  2.74s/it]
Epoch Progress:  10%|██▍                      | 96/1000 [04:26<41:25,  2.75s/it]
Epoch Progress:  10%|██▍                      | 97/1000 [04:29<41:27,  2.75s/it]
Epoch Progress:  10%|██▍                      | 98/1000 [04:32<41:32,  2.76s/it]
Epoch Progress:  10%|██▍                      | 99/1000 [04:34<41:22,  2.76s/it]
Epoch Progress:  10%|██▍                     | 100/1000 [04:37<41:16,  2.75s/it]Dataset Size: 10%, Epoch: 101, Train Loss: 4.884461239764565, Val Loss: 7.545833673232641

Epoch Progress:  10%|██▍                     | 101/1000 [04:40<41:14,  2.75s/it]
Epoch Progress:  10%|██▍                     | 102/1000 [04:43<41:10,  2.75s/it]
Epoch Progress:  10%|██▍                     | 103/1000 [04:45<41:09,  2.75s/it]
Epoch Progress:  10%|██▍                     | 104/1000 [04:48<41:03,  2.75s/it]
Epoch Progress:  10%|██▌                     | 105/1000 [04:51<41:02,  2.75s/it]
Epoch Progress:  11%|██▌                     | 106/1000 [04:54<41:03,  2.76s/it]
Epoch Progress:  11%|██▌                     | 107/1000 [04:56<40:57,  2.75s/it]
Epoch Progress:  11%|██▌                     | 108/1000 [04:59<40:56,  2.75s/it]
Epoch Progress:  11%|██▌                     | 109/1000 [05:02<41:01,  2.76s/it]
Epoch Progress:  11%|██▋                     | 110/1000 [05:05<40:54,  2.76s/it]Dataset Size: 10%, Epoch: 111, Train Loss: 4.699278266806352, Val Loss: 7.706703748458471

Epoch Progress:  11%|██▋                     | 111/1000 [05:07<40:50,  2.76s/it]
Epoch Progress:  11%|██▋                     | 112/1000 [05:10<40:48,  2.76s/it]
Epoch Progress:  11%|██▋                     | 113/1000 [05:13<40:45,  2.76s/it]
Epoch Progress:  11%|██▋                     | 114/1000 [05:16<40:45,  2.76s/it]
Epoch Progress:  12%|██▊                     | 115/1000 [05:18<40:42,  2.76s/it]
Epoch Progress:  12%|██▊                     | 116/1000 [05:21<40:42,  2.76s/it]
Epoch Progress:  12%|██▊                     | 117/1000 [05:24<40:35,  2.76s/it]
Epoch Progress:  12%|██▊                     | 118/1000 [05:27<40:30,  2.76s/it]
Epoch Progress:  12%|██▊                     | 119/1000 [05:29<40:28,  2.76s/it]
Epoch Progress:  12%|██▉                     | 120/1000 [05:32<40:32,  2.76s/it]Dataset Size: 10%, Epoch: 121, Train Loss: 4.559916922920628, Val Loss: 7.902143087142553

Epoch Progress:  12%|██▉                     | 121/1000 [05:35<40:31,  2.77s/it]
Epoch Progress:  12%|██▉                     | 122/1000 [05:38<40:26,  2.76s/it]
Epoch Progress:  12%|██▉                     | 123/1000 [05:41<40:24,  2.77s/it]
Epoch Progress:  12%|██▉                     | 124/1000 [05:43<40:25,  2.77s/it]
Epoch Progress:  12%|███                     | 125/1000 [05:46<40:24,  2.77s/it]
Epoch Progress:  13%|███                     | 126/1000 [05:49<40:18,  2.77s/it]
Epoch Progress:  13%|███                     | 127/1000 [05:52<40:16,  2.77s/it]
Epoch Progress:  13%|███                     | 128/1000 [05:54<40:10,  2.76s/it]
Epoch Progress:  13%|███                     | 129/1000 [05:57<40:00,  2.76s/it]
Epoch Progress:  13%|███                     | 130/1000 [06:00<39:55,  2.75s/it]Dataset Size: 10%, Epoch: 131, Train Loss: 4.449413362302278, Val Loss: 8.079998358702047

Epoch Progress:  13%|███▏                    | 131/1000 [06:03<39:51,  2.75s/it]
Epoch Progress:  13%|███▏                    | 132/1000 [06:05<39:50,  2.75s/it]
Epoch Progress:  13%|███▏                    | 133/1000 [06:08<39:46,  2.75s/it]
Epoch Progress:  13%|███▏                    | 134/1000 [06:11<39:47,  2.76s/it]
Epoch Progress:  14%|███▏                    | 135/1000 [06:14<39:41,  2.75s/it]
Epoch Progress:  14%|███▎                    | 136/1000 [06:16<39:34,  2.75s/it]
Epoch Progress:  14%|███▎                    | 137/1000 [06:19<39:29,  2.75s/it]
Epoch Progress:  14%|███▎                    | 138/1000 [06:22<39:28,  2.75s/it]
Epoch Progress:  14%|███▎                    | 139/1000 [06:25<39:24,  2.75s/it]
Epoch Progress:  14%|███▎                    | 140/1000 [06:27<39:18,  2.74s/it]Dataset Size: 10%, Epoch: 141, Train Loss: 4.374131441116333, Val Loss: 8.250718569144224

Epoch Progress:  14%|███▍                    | 141/1000 [06:30<39:15,  2.74s/it]
Epoch Progress:  14%|███▍                    | 142/1000 [06:33<39:13,  2.74s/it]
Epoch Progress:  14%|███▍                    | 143/1000 [06:36<39:09,  2.74s/it]
Epoch Progress:  14%|███▍                    | 144/1000 [06:38<39:08,  2.74s/it]
Epoch Progress:  14%|███▍                    | 145/1000 [06:41<39:06,  2.74s/it]
Epoch Progress:  15%|███▌                    | 146/1000 [06:44<39:03,  2.74s/it]
Epoch Progress:  15%|███▌                    | 147/1000 [06:47<38:58,  2.74s/it]
Epoch Progress:  15%|███▌                    | 148/1000 [06:49<38:54,  2.74s/it]
Epoch Progress:  15%|███▌                    | 149/1000 [06:52<38:53,  2.74s/it]
Epoch Progress:  15%|███▌                    | 150/1000 [06:55<38:48,  2.74s/it]Dataset Size: 10%, Epoch: 151, Train Loss: 4.288957507986772, Val Loss: 8.410241664984287

Epoch Progress:  15%|███▌                    | 151/1000 [06:57<38:47,  2.74s/it]
Epoch Progress:  15%|███▋                    | 152/1000 [07:00<38:49,  2.75s/it]
Epoch Progress:  15%|███▋                    | 153/1000 [07:03<38:46,  2.75s/it]
Epoch Progress:  15%|███▋                    | 154/1000 [07:06<38:40,  2.74s/it]
Epoch Progress:  16%|███▋                    | 155/1000 [07:08<38:40,  2.75s/it]
Epoch Progress:  16%|███▋                    | 156/1000 [07:11<38:38,  2.75s/it]
Epoch Progress:  16%|███▊                    | 157/1000 [07:14<38:29,  2.74s/it]
Epoch Progress:  16%|███▊                    | 158/1000 [07:17<38:24,  2.74s/it]
Epoch Progress:  16%|███▊                    | 159/1000 [07:19<38:22,  2.74s/it]
Epoch Progress:  16%|███▊                    | 160/1000 [07:22<38:19,  2.74s/it]Dataset Size: 10%, Epoch: 161, Train Loss: 4.236599081440976, Val Loss: 8.587388258713942

Epoch Progress:  16%|███▊                    | 161/1000 [07:25<38:14,  2.74s/it]
Epoch Progress:  16%|███▉                    | 162/1000 [07:28<38:09,  2.73s/it]
Epoch Progress:  16%|███▉                    | 163/1000 [07:30<38:06,  2.73s/it]
Epoch Progress:  16%|███▉                    | 164/1000 [07:33<38:04,  2.73s/it]
Epoch Progress:  16%|███▉                    | 165/1000 [07:36<37:58,  2.73s/it]
Epoch Progress:  17%|███▉                    | 166/1000 [07:39<37:56,  2.73s/it]
Epoch Progress:  17%|████                    | 167/1000 [07:41<37:58,  2.73s/it]
Epoch Progress:  17%|████                    | 168/1000 [07:44<37:53,  2.73s/it]
Epoch Progress:  17%|████                    | 169/1000 [07:47<37:48,  2.73s/it]
Epoch Progress:  17%|████                    | 170/1000 [07:49<37:46,  2.73s/it]Dataset Size: 10%, Epoch: 171, Train Loss: 4.188599611583509, Val Loss: 8.746108201833872

Epoch Progress:  17%|████                    | 171/1000 [07:52<37:42,  2.73s/it]
Epoch Progress:  17%|████▏                   | 172/1000 [07:55<37:39,  2.73s/it]
Epoch Progress:  17%|████▏                   | 173/1000 [07:58<37:36,  2.73s/it]
Epoch Progress:  17%|████▏                   | 174/1000 [08:00<37:33,  2.73s/it]
Epoch Progress:  18%|████▏                   | 175/1000 [08:03<37:32,  2.73s/it]
Epoch Progress:  18%|████▏                   | 176/1000 [08:06<37:30,  2.73s/it]
Epoch Progress:  18%|████▏                   | 177/1000 [08:09<37:28,  2.73s/it]
Epoch Progress:  18%|████▎                   | 178/1000 [08:11<37:33,  2.74s/it]
Epoch Progress:  18%|████▎                   | 179/1000 [08:14<37:28,  2.74s/it]
Epoch Progress:  18%|████▎                   | 180/1000 [08:17<37:27,  2.74s/it]Dataset Size: 10%, Epoch: 181, Train Loss: 4.151577585621884, Val Loss: 8.908896886385405

Epoch Progress:  18%|████▎                   | 181/1000 [08:20<37:26,  2.74s/it]
Epoch Progress:  18%|████▎                   | 182/1000 [08:22<37:29,  2.75s/it]
Epoch Progress:  18%|████▍                   | 183/1000 [08:25<37:28,  2.75s/it]
Epoch Progress:  18%|████▍                   | 184/1000 [08:28<37:24,  2.75s/it]
Epoch Progress:  18%|████▍                   | 185/1000 [08:31<37:23,  2.75s/it]
Epoch Progress:  19%|████▍                   | 186/1000 [08:33<37:19,  2.75s/it]
Epoch Progress:  19%|████▍                   | 187/1000 [08:36<37:13,  2.75s/it]
Epoch Progress:  19%|████▌                   | 188/1000 [08:39<37:11,  2.75s/it]
Epoch Progress:  19%|████▌                   | 189/1000 [08:42<37:09,  2.75s/it]
Epoch Progress:  19%|████▌                   | 190/1000 [08:44<37:11,  2.76s/it]Dataset Size: 10%, Epoch: 191, Train Loss: 4.111332109099941, Val Loss: 9.052464167277018

Epoch Progress:  19%|████▌                   | 191/1000 [08:47<37:08,  2.75s/it]
Epoch Progress:  19%|████▌                   | 192/1000 [08:50<37:07,  2.76s/it]
Epoch Progress:  19%|████▋                   | 193/1000 [08:53<37:06,  2.76s/it]
Epoch Progress:  19%|████▋                   | 194/1000 [08:55<37:06,  2.76s/it]
Epoch Progress:  20%|████▋                   | 195/1000 [08:58<36:58,  2.76s/it]
Epoch Progress:  20%|████▋                   | 196/1000 [09:01<36:57,  2.76s/it]
Epoch Progress:  20%|████▋                   | 197/1000 [09:04<36:56,  2.76s/it]
Epoch Progress:  20%|████▊                   | 198/1000 [09:06<36:54,  2.76s/it]
Epoch Progress:  20%|████▊                   | 199/1000 [09:09<36:53,  2.76s/it]
Epoch Progress:  20%|████▊                   | 200/1000 [09:12<36:53,  2.77s/it]Dataset Size: 10%, Epoch: 201, Train Loss: 4.087084519235711, Val Loss: 9.179111505166079

Epoch Progress:  20%|████▊                   | 201/1000 [09:15<36:55,  2.77s/it]
Epoch Progress:  20%|████▊                   | 202/1000 [09:18<36:51,  2.77s/it]
Epoch Progress:  20%|████▊                   | 203/1000 [09:20<36:42,  2.76s/it]
Epoch Progress:  20%|████▉                   | 204/1000 [09:23<36:40,  2.76s/it]
Epoch Progress:  20%|████▉                   | 205/1000 [09:26<36:40,  2.77s/it]
Epoch Progress:  21%|████▉                   | 206/1000 [09:29<36:35,  2.76s/it]
Epoch Progress:  21%|████▉                   | 207/1000 [09:31<36:33,  2.77s/it]
Epoch Progress:  21%|████▉                   | 208/1000 [09:34<36:31,  2.77s/it]
Epoch Progress:  21%|█████                   | 209/1000 [09:37<36:26,  2.76s/it]
Epoch Progress:  21%|█████                   | 210/1000 [09:40<36:26,  2.77s/it]Dataset Size: 10%, Epoch: 211, Train Loss: 4.050162842399196, Val Loss: 9.345243820777306

Epoch Progress:  21%|█████                   | 211/1000 [09:42<36:23,  2.77s/it]
Epoch Progress:  21%|█████                   | 212/1000 [09:45<36:19,  2.77s/it]
Epoch Progress:  21%|█████                   | 213/1000 [09:48<36:18,  2.77s/it]
Epoch Progress:  21%|█████▏                  | 214/1000 [09:51<36:16,  2.77s/it]
Epoch Progress:  22%|█████▏                  | 215/1000 [09:53<36:14,  2.77s/it]
Epoch Progress:  22%|█████▏                  | 216/1000 [09:56<36:11,  2.77s/it]
Epoch Progress:  22%|█████▏                  | 217/1000 [09:59<36:06,  2.77s/it]
Epoch Progress:  22%|█████▏                  | 218/1000 [10:02<36:05,  2.77s/it]
Epoch Progress:  22%|█████▎                  | 219/1000 [10:05<36:01,  2.77s/it]
Epoch Progress:  22%|█████▎                  | 220/1000 [10:07<35:58,  2.77s/it]Dataset Size: 10%, Epoch: 221, Train Loss: 4.023785860914933, Val Loss: 9.48945150619898

Epoch Progress:  22%|█████▎                  | 221/1000 [10:10<35:53,  2.77s/it]
Epoch Progress:  22%|█████▎                  | 222/1000 [10:13<35:52,  2.77s/it]
Epoch Progress:  22%|█████▎                  | 223/1000 [10:16<35:49,  2.77s/it]
Epoch Progress:  22%|█████▍                  | 224/1000 [10:18<35:50,  2.77s/it]
Epoch Progress:  22%|█████▍                  | 225/1000 [10:21<35:48,  2.77s/it]
Epoch Progress:  23%|█████▍                  | 226/1000 [10:24<35:44,  2.77s/it]
Epoch Progress:  23%|█████▍                  | 227/1000 [10:27<35:39,  2.77s/it]
Epoch Progress:  23%|█████▍                  | 228/1000 [10:29<35:35,  2.77s/it]
Epoch Progress:  23%|█████▍                  | 229/1000 [10:32<35:35,  2.77s/it]
Epoch Progress:  23%|█████▌                  | 230/1000 [10:35<35:31,  2.77s/it]Dataset Size: 10%, Epoch: 231, Train Loss: 3.994344485433478, Val Loss: 9.637772535666441

Epoch Progress:  23%|█████▌                  | 231/1000 [10:38<35:25,  2.76s/it]
Epoch Progress:  23%|█████▌                  | 232/1000 [10:41<35:21,  2.76s/it]
Epoch Progress:  23%|█████▌                  | 233/1000 [10:43<35:20,  2.76s/it]
Epoch Progress:  23%|█████▌                  | 234/1000 [10:46<35:18,  2.77s/it]
Epoch Progress:  24%|█████▋                  | 235/1000 [10:49<35:14,  2.76s/it]
Epoch Progress:  24%|█████▋                  | 236/1000 [10:52<35:17,  2.77s/it]
Epoch Progress:  24%|█████▋                  | 237/1000 [10:54<35:10,  2.77s/it]
Epoch Progress:  24%|█████▋                  | 238/1000 [10:57<35:06,  2.76s/it]
Epoch Progress:  24%|█████▋                  | 239/1000 [11:00<35:03,  2.76s/it]
Epoch Progress:  24%|█████▊                  | 240/1000 [11:03<35:00,  2.76s/it]Dataset Size: 10%, Epoch: 241, Train Loss: 3.9862528976641203, Val Loss: 9.777890669993866

Epoch Progress:  24%|█████▊                  | 241/1000 [11:05<34:56,  2.76s/it]
Epoch Progress:  24%|█████▊                  | 242/1000 [11:08<34:58,  2.77s/it]
Epoch Progress:  24%|█████▊                  | 243/1000 [11:11<34:59,  2.77s/it]
Epoch Progress:  24%|█████▊                  | 244/1000 [11:14<34:57,  2.77s/it]
Epoch Progress:  24%|█████▉                  | 245/1000 [11:17<34:55,  2.77s/it]
Epoch Progress:  25%|█████▉                  | 246/1000 [11:19<34:53,  2.78s/it]
Epoch Progress:  25%|█████▉                  | 247/1000 [11:22<34:57,  2.79s/it]
Epoch Progress:  25%|█████▉                  | 248/1000 [11:25<34:48,  2.78s/it]
Epoch Progress:  25%|█████▉                  | 249/1000 [11:28<34:43,  2.77s/it]
Epoch Progress:  25%|██████                  | 250/1000 [11:30<34:39,  2.77s/it]Dataset Size: 10%, Epoch: 251, Train Loss: 3.9583327205557572, Val Loss: 9.90923959780962

Epoch Progress:  25%|██████                  | 251/1000 [11:33<34:36,  2.77s/it]
Epoch Progress:  25%|██████                  | 252/1000 [11:36<34:31,  2.77s/it]
Epoch Progress:  25%|██████                  | 253/1000 [11:39<34:29,  2.77s/it]
Epoch Progress:  25%|██████                  | 254/1000 [11:41<34:28,  2.77s/it]
Epoch Progress:  26%|██████                  | 255/1000 [11:44<34:25,  2.77s/it]
Epoch Progress:  26%|██████▏                 | 256/1000 [11:47<34:20,  2.77s/it]
Epoch Progress:  26%|██████▏                 | 257/1000 [11:50<34:17,  2.77s/it]
Epoch Progress:  26%|██████▏                 | 258/1000 [11:53<34:12,  2.77s/it]
Epoch Progress:  26%|██████▏                 | 259/1000 [11:55<34:12,  2.77s/it]
Epoch Progress:  26%|██████▏                 | 260/1000 [11:58<34:06,  2.77s/it]Dataset Size: 10%, Epoch: 261, Train Loss: 3.9412491886239303, Val Loss: 10.044854286389473

Epoch Progress:  26%|██████▎                 | 261/1000 [12:01<34:04,  2.77s/it]
Epoch Progress:  26%|██████▎                 | 262/1000 [12:04<34:02,  2.77s/it]
Epoch Progress:  26%|██████▎                 | 263/1000 [12:06<33:56,  2.76s/it]
Epoch Progress:  26%|██████▎                 | 264/1000 [12:09<33:53,  2.76s/it]
Epoch Progress:  26%|██████▎                 | 265/1000 [12:12<33:53,  2.77s/it]
Epoch Progress:  27%|██████▍                 | 266/1000 [12:15<33:49,  2.77s/it]
Epoch Progress:  27%|██████▍                 | 267/1000 [12:17<33:46,  2.76s/it]
Epoch Progress:  27%|██████▍                 | 268/1000 [12:20<33:39,  2.76s/it]
Epoch Progress:  27%|██████▍                 | 269/1000 [12:23<33:37,  2.76s/it]
Epoch Progress:  27%|██████▍                 | 270/1000 [12:26<33:39,  2.77s/it]Dataset Size: 10%, Epoch: 271, Train Loss: 3.925349153970417, Val Loss: 10.159158780024601

Epoch Progress:  27%|██████▌                 | 271/1000 [12:28<33:34,  2.76s/it]
Epoch Progress:  27%|██████▌                 | 272/1000 [12:31<33:37,  2.77s/it]
Epoch Progress:  27%|██████▌                 | 273/1000 [12:34<33:32,  2.77s/it]
Epoch Progress:  27%|██████▌                 | 274/1000 [12:37<33:29,  2.77s/it]
Epoch Progress:  28%|██████▌                 | 275/1000 [12:40<33:24,  2.76s/it]
Epoch Progress:  28%|██████▌                 | 276/1000 [12:42<33:22,  2.77s/it]
Epoch Progress:  28%|██████▋                 | 277/1000 [12:45<33:18,  2.76s/it]
Epoch Progress:  28%|██████▋                 | 278/1000 [12:48<33:12,  2.76s/it]
Epoch Progress:  28%|██████▋                 | 279/1000 [12:51<33:11,  2.76s/it]
Epoch Progress:  28%|██████▋                 | 280/1000 [12:53<33:07,  2.76s/it]Dataset Size: 10%, Epoch: 281, Train Loss: 3.910122231433266, Val Loss: 10.290386444483048

Epoch Progress:  28%|██████▋                 | 281/1000 [12:56<33:06,  2.76s/it]
Epoch Progress:  28%|██████▊                 | 282/1000 [12:59<33:04,  2.76s/it]
Epoch Progress:  28%|██████▊                 | 283/1000 [13:02<33:03,  2.77s/it]
Epoch Progress:  28%|██████▊                 | 284/1000 [13:04<32:58,  2.76s/it]
Epoch Progress:  28%|██████▊                 | 285/1000 [13:07<32:53,  2.76s/it]
Epoch Progress:  29%|██████▊                 | 286/1000 [13:10<32:50,  2.76s/it]
Epoch Progress:  29%|██████▉                 | 287/1000 [13:13<32:50,  2.76s/it]
Epoch Progress:  29%|██████▉                 | 288/1000 [13:15<32:47,  2.76s/it]
Epoch Progress:  29%|██████▉                 | 289/1000 [13:18<32:44,  2.76s/it]
Epoch Progress:  29%|██████▉                 | 290/1000 [13:21<32:43,  2.77s/it]Dataset Size: 10%, Epoch: 291, Train Loss: 3.901531846899735, Val Loss: 10.416988690694174

Epoch Progress:  29%|██████▉                 | 291/1000 [13:24<32:40,  2.76s/it]
Epoch Progress:  29%|███████                 | 292/1000 [13:27<32:35,  2.76s/it]
Epoch Progress:  29%|███████                 | 293/1000 [13:29<32:38,  2.77s/it]
Epoch Progress:  29%|███████                 | 294/1000 [13:32<32:37,  2.77s/it]
Epoch Progress:  30%|███████                 | 295/1000 [13:35<32:29,  2.77s/it]
Epoch Progress:  30%|███████                 | 296/1000 [13:38<32:26,  2.76s/it]
Epoch Progress:  30%|███████▏                | 297/1000 [13:40<32:24,  2.77s/it]
Epoch Progress:  30%|███████▏                | 298/1000 [13:43<32:19,  2.76s/it]
Epoch Progress:  30%|███████▏                | 299/1000 [13:46<32:12,  2.76s/it]
Epoch Progress:  30%|███████▏                | 300/1000 [13:49<32:11,  2.76s/it]Dataset Size: 10%, Epoch: 301, Train Loss: 3.8797698962061027, Val Loss: 10.541200026487692

Epoch Progress:  30%|███████▏                | 301/1000 [13:51<32:10,  2.76s/it]
Epoch Progress:  30%|███████▏                | 302/1000 [13:54<32:07,  2.76s/it]
Epoch Progress:  30%|███████▎                | 303/1000 [13:57<32:03,  2.76s/it]
Epoch Progress:  30%|███████▎                | 304/1000 [14:00<32:03,  2.76s/it]
Epoch Progress:  30%|███████▎                | 305/1000 [14:02<32:06,  2.77s/it]
Epoch Progress:  31%|███████▎                | 306/1000 [14:05<32:03,  2.77s/it]
Epoch Progress:  31%|███████▎                | 307/1000 [14:08<32:00,  2.77s/it]
Epoch Progress:  31%|███████▍                | 308/1000 [14:11<31:59,  2.77s/it]
Epoch Progress:  31%|███████▍                | 309/1000 [14:14<31:53,  2.77s/it]
Epoch Progress:  31%|███████▍                | 310/1000 [14:16<31:46,  2.76s/it]Dataset Size: 10%, Epoch: 311, Train Loss: 3.871660307834023, Val Loss: 10.660346984863281

Epoch Progress:  31%|███████▍                | 311/1000 [14:19<31:46,  2.77s/it]
Epoch Progress:  31%|███████▍                | 312/1000 [14:22<31:44,  2.77s/it]
Epoch Progress:  31%|███████▌                | 313/1000 [14:25<31:39,  2.76s/it]
Epoch Progress:  31%|███████▌                | 314/1000 [14:27<31:37,  2.77s/it]
Epoch Progress:  32%|███████▌                | 315/1000 [14:30<31:34,  2.76s/it]
Epoch Progress:  32%|███████▌                | 316/1000 [14:33<31:35,  2.77s/it]
Epoch Progress:  32%|███████▌                | 317/1000 [14:36<31:31,  2.77s/it]
Epoch Progress:  32%|███████▋                | 318/1000 [14:38<31:29,  2.77s/it]
Epoch Progress:  32%|███████▋                | 319/1000 [14:41<31:26,  2.77s/it]
Epoch Progress:  32%|███████▋                | 320/1000 [14:44<31:19,  2.76s/it]Dataset Size: 10%, Epoch: 321, Train Loss: 3.8563600402129325, Val Loss: 10.764638485052647

Epoch Progress:  32%|███████▋                | 321/1000 [14:47<31:16,  2.76s/it]
Epoch Progress:  32%|███████▋                | 322/1000 [14:50<31:13,  2.76s/it]
Epoch Progress:  32%|███████▊                | 323/1000 [14:52<31:10,  2.76s/it]
Epoch Progress:  32%|███████▊                | 324/1000 [14:55<31:06,  2.76s/it]
Epoch Progress:  32%|███████▊                | 325/1000 [14:58<31:04,  2.76s/it]
Epoch Progress:  33%|███████▊                | 326/1000 [15:01<31:01,  2.76s/it]
Epoch Progress:  33%|███████▊                | 327/1000 [15:03<31:03,  2.77s/it]
Epoch Progress:  33%|███████▊                | 328/1000 [15:06<31:01,  2.77s/it]
Epoch Progress:  33%|███████▉                | 329/1000 [15:09<30:58,  2.77s/it]
Epoch Progress:  33%|███████▉                | 330/1000 [15:12<30:54,  2.77s/it]Dataset Size: 10%, Epoch: 331, Train Loss: 3.841979296583878, Val Loss: 10.874888958075108

Epoch Progress:  33%|███████▉                | 331/1000 [15:14<30:49,  2.76s/it]
Epoch Progress:  33%|███████▉                | 332/1000 [15:17<30:46,  2.76s/it]
Epoch Progress:  33%|███████▉                | 333/1000 [15:20<30:42,  2.76s/it]
Epoch Progress:  33%|████████                | 334/1000 [15:23<30:40,  2.76s/it]
Epoch Progress:  34%|████████                | 335/1000 [15:25<30:37,  2.76s/it]
Epoch Progress:  34%|████████                | 336/1000 [15:28<30:33,  2.76s/it]
Epoch Progress:  34%|████████                | 337/1000 [15:31<30:33,  2.77s/it]
Epoch Progress:  34%|████████                | 338/1000 [15:34<30:30,  2.77s/it]
Epoch Progress:  34%|████████▏               | 339/1000 [15:37<30:29,  2.77s/it]
Epoch Progress:  34%|████████▏               | 340/1000 [15:39<30:23,  2.76s/it]Dataset Size: 10%, Epoch: 341, Train Loss: 3.8289013724578056, Val Loss: 10.984415421119103

Epoch Progress:  34%|████████▏               | 341/1000 [15:42<30:22,  2.77s/it]
Epoch Progress:  34%|████████▏               | 342/1000 [15:45<30:19,  2.77s/it]
Epoch Progress:  34%|████████▏               | 343/1000 [15:48<30:16,  2.77s/it]
Epoch Progress:  34%|████████▎               | 344/1000 [15:50<30:13,  2.76s/it]
Epoch Progress:  34%|████████▎               | 345/1000 [15:53<30:09,  2.76s/it]
Epoch Progress:  35%|████████▎               | 346/1000 [15:56<30:08,  2.77s/it]
Epoch Progress:  35%|████████▎               | 347/1000 [15:59<30:03,  2.76s/it]
Epoch Progress:  35%|████████▎               | 348/1000 [16:01<30:02,  2.77s/it]
Epoch Progress:  35%|████████▍               | 349/1000 [16:04<29:59,  2.76s/it]
Epoch Progress:  35%|████████▍               | 350/1000 [16:07<30:01,  2.77s/it]Dataset Size: 10%, Epoch: 351, Train Loss: 3.825889675240768, Val Loss: 11.071432651617588

Epoch Progress:  35%|████████▍               | 351/1000 [16:10<29:57,  2.77s/it]
Epoch Progress:  35%|████████▍               | 352/1000 [16:12<29:53,  2.77s/it]
Epoch Progress:  35%|████████▍               | 353/1000 [16:15<29:48,  2.76s/it]
Epoch Progress:  35%|████████▍               | 354/1000 [16:18<29:45,  2.76s/it]
Epoch Progress:  36%|████████▌               | 355/1000 [16:21<29:43,  2.77s/it]
Epoch Progress:  36%|████████▌               | 356/1000 [16:24<29:42,  2.77s/it]
Epoch Progress:  36%|████████▌               | 357/1000 [16:26<29:38,  2.77s/it]
Epoch Progress:  36%|████████▌               | 358/1000 [16:29<29:34,  2.76s/it]
Epoch Progress:  36%|████████▌               | 359/1000 [16:32<29:33,  2.77s/it]
Epoch Progress:  36%|████████▋               | 360/1000 [16:35<29:28,  2.76s/it]Dataset Size: 10%, Epoch: 361, Train Loss: 3.8099847279096903, Val Loss: 11.177814752627642

Epoch Progress:  36%|████████▋               | 361/1000 [16:37<29:24,  2.76s/it]
Epoch Progress:  36%|████████▋               | 362/1000 [16:40<29:27,  2.77s/it]
Epoch Progress:  36%|████████▋               | 363/1000 [16:43<29:25,  2.77s/it]
Epoch Progress:  36%|████████▋               | 364/1000 [16:46<29:23,  2.77s/it]
Epoch Progress:  36%|████████▊               | 365/1000 [16:48<29:23,  2.78s/it]
Epoch Progress:  37%|████████▊               | 366/1000 [16:51<29:22,  2.78s/it]
Epoch Progress:  37%|████████▊               | 367/1000 [16:54<29:18,  2.78s/it]
Epoch Progress:  37%|████████▊               | 368/1000 [16:57<29:14,  2.78s/it]
Epoch Progress:  37%|████████▊               | 369/1000 [17:00<29:10,  2.77s/it]
Epoch Progress:  37%|████████▉               | 370/1000 [17:02<29:05,  2.77s/it]Dataset Size: 10%, Epoch: 371, Train Loss: 3.7991261795947424, Val Loss: 11.256795125129896

Epoch Progress:  37%|████████▉               | 371/1000 [17:05<29:00,  2.77s/it]
Epoch Progress:  37%|████████▉               | 372/1000 [17:08<28:56,  2.77s/it]
Epoch Progress:  37%|████████▉               | 373/1000 [17:11<29:01,  2.78s/it]
Epoch Progress:  37%|████████▉               | 374/1000 [17:13<28:57,  2.78s/it]
Epoch Progress:  38%|█████████               | 375/1000 [17:16<28:51,  2.77s/it]
Epoch Progress:  38%|█████████               | 376/1000 [17:19<28:49,  2.77s/it]
Epoch Progress:  38%|█████████               | 377/1000 [17:22<28:46,  2.77s/it]
Epoch Progress:  38%|█████████               | 378/1000 [17:25<28:43,  2.77s/it]
Epoch Progress:  38%|█████████               | 379/1000 [17:27<28:38,  2.77s/it]
Epoch Progress:  38%|█████████               | 380/1000 [17:30<28:33,  2.76s/it]Dataset Size: 10%, Epoch: 381, Train Loss: 3.7993205032850565, Val Loss: 11.365475703508427

Epoch Progress:  38%|█████████▏              | 381/1000 [17:33<28:33,  2.77s/it]
Epoch Progress:  38%|█████████▏              | 382/1000 [17:36<28:27,  2.76s/it]
Epoch Progress:  38%|█████████▏              | 383/1000 [17:38<28:23,  2.76s/it]
Epoch Progress:  38%|█████████▏              | 384/1000 [17:41<28:22,  2.76s/it]
Epoch Progress:  38%|█████████▏              | 385/1000 [17:44<28:22,  2.77s/it]
Epoch Progress:  39%|█████████▎              | 386/1000 [17:47<28:17,  2.76s/it]
Epoch Progress:  39%|█████████▎              | 387/1000 [17:49<28:12,  2.76s/it]
Epoch Progress:  39%|█████████▎              | 388/1000 [17:52<28:11,  2.76s/it]
Epoch Progress:  39%|█████████▎              | 389/1000 [17:55<28:09,  2.76s/it]
Epoch Progress:  39%|█████████▎              | 390/1000 [17:58<28:04,  2.76s/it]Dataset Size: 10%, Epoch: 391, Train Loss: 3.7907921389529577, Val Loss: 11.448058054997372

Epoch Progress:  39%|█████████▍              | 391/1000 [18:00<28:03,  2.76s/it]
Epoch Progress:  39%|█████████▍              | 392/1000 [18:03<28:00,  2.76s/it]
Epoch Progress:  39%|█████████▍              | 393/1000 [18:06<27:58,  2.77s/it]
Epoch Progress:  39%|█████████▍              | 394/1000 [18:09<27:52,  2.76s/it]
Epoch Progress:  40%|█████████▍              | 395/1000 [18:11<27:51,  2.76s/it]
Epoch Progress:  40%|█████████▌              | 396/1000 [18:14<27:54,  2.77s/it]
Epoch Progress:  40%|█████████▌              | 397/1000 [18:17<27:46,  2.76s/it]
Epoch Progress:  40%|█████████▌              | 398/1000 [18:20<27:41,  2.76s/it]
Epoch Progress:  40%|█████████▌              | 399/1000 [18:23<27:41,  2.76s/it]
Epoch Progress:  40%|█████████▌              | 400/1000 [18:25<27:38,  2.76s/it]Dataset Size: 10%, Epoch: 401, Train Loss: 3.7803974841770374, Val Loss: 11.54889617822109

Epoch Progress:  40%|█████████▌              | 401/1000 [18:28<27:33,  2.76s/it]
Epoch Progress:  40%|█████████▋              | 402/1000 [18:31<27:36,  2.77s/it]
Epoch Progress:  40%|█████████▋              | 403/1000 [18:34<27:31,  2.77s/it]
Epoch Progress:  40%|█████████▋              | 404/1000 [18:36<27:26,  2.76s/it]
Epoch Progress:  40%|█████████▋              | 405/1000 [18:39<27:22,  2.76s/it]
Epoch Progress:  41%|█████████▋              | 406/1000 [18:42<27:22,  2.76s/it]
Epoch Progress:  41%|█████████▊              | 407/1000 [18:45<27:17,  2.76s/it]
Epoch Progress:  41%|█████████▊              | 408/1000 [18:47<27:13,  2.76s/it]
Epoch Progress:  41%|█████████▊              | 409/1000 [18:50<27:12,  2.76s/it]
Epoch Progress:  41%|█████████▊              | 410/1000 [18:53<27:10,  2.76s/it]Dataset Size: 10%, Epoch: 411, Train Loss: 3.7771712855288855, Val Loss: 11.624679247538248

Epoch Progress:  41%|█████████▊              | 411/1000 [18:56<27:07,  2.76s/it]
Epoch Progress:  41%|█████████▉              | 412/1000 [18:58<27:01,  2.76s/it]
Epoch Progress:  41%|█████████▉              | 413/1000 [19:01<27:02,  2.76s/it]
Epoch Progress:  41%|█████████▉              | 414/1000 [19:04<27:00,  2.76s/it]
Epoch Progress:  42%|█████████▉              | 415/1000 [19:07<26:55,  2.76s/it]
Epoch Progress:  42%|█████████▉              | 416/1000 [19:09<26:52,  2.76s/it]
Epoch Progress:  42%|██████████              | 417/1000 [19:12<26:50,  2.76s/it]
Epoch Progress:  42%|██████████              | 418/1000 [19:15<26:48,  2.76s/it]
Epoch Progress:  42%|██████████              | 419/1000 [19:18<26:48,  2.77s/it]
Epoch Progress:  42%|██████████              | 420/1000 [19:21<26:46,  2.77s/it]Dataset Size: 10%, Epoch: 421, Train Loss: 3.7643427284140336, Val Loss: 11.724677526033842

Epoch Progress:  42%|██████████              | 421/1000 [19:23<26:44,  2.77s/it]
Epoch Progress:  42%|██████████▏             | 422/1000 [19:26<26:38,  2.77s/it]
Epoch Progress:  42%|██████████▏             | 423/1000 [19:29<26:35,  2.77s/it]
Epoch Progress:  42%|██████████▏             | 424/1000 [19:32<26:35,  2.77s/it]
Epoch Progress:  42%|██████████▏             | 425/1000 [19:34<26:33,  2.77s/it]
Epoch Progress:  43%|██████████▏             | 426/1000 [19:37<26:31,  2.77s/it]
Epoch Progress:  43%|██████████▏             | 427/1000 [19:40<26:29,  2.77s/it]
Epoch Progress:  43%|██████████▎             | 428/1000 [19:43<26:27,  2.78s/it]
Epoch Progress:  43%|██████████▎             | 429/1000 [19:46<26:24,  2.77s/it]
Epoch Progress:  43%|██████████▎             | 430/1000 [19:48<26:22,  2.78s/it]Dataset Size: 10%, Epoch: 431, Train Loss: 3.755743516118903, Val Loss: 11.808704058329264

Epoch Progress:  43%|██████████▎             | 431/1000 [19:51<26:23,  2.78s/it]
Epoch Progress:  43%|██████████▎             | 432/1000 [19:54<26:18,  2.78s/it]
Epoch Progress:  43%|██████████▍             | 433/1000 [19:57<26:11,  2.77s/it]
Epoch Progress:  43%|██████████▍             | 434/1000 [19:59<26:06,  2.77s/it]
Epoch Progress:  44%|██████████▍             | 435/1000 [20:02<26:02,  2.77s/it]
Epoch Progress:  44%|██████████▍             | 436/1000 [20:05<25:57,  2.76s/it]
Epoch Progress:  44%|██████████▍             | 437/1000 [20:08<25:53,  2.76s/it]
Epoch Progress:  44%|██████████▌             | 438/1000 [20:10<25:50,  2.76s/it]
Epoch Progress:  44%|██████████▌             | 439/1000 [20:13<25:49,  2.76s/it]
Epoch Progress:  44%|██████████▌             | 440/1000 [20:16<25:44,  2.76s/it]Dataset Size: 10%, Epoch: 441, Train Loss: 3.75299828930905, Val Loss: 11.87630760975373

Epoch Progress:  44%|██████████▌             | 441/1000 [20:19<25:42,  2.76s/it]
Epoch Progress:  44%|██████████▌             | 442/1000 [20:21<25:44,  2.77s/it]
Epoch Progress:  44%|██████████▋             | 443/1000 [20:24<25:40,  2.77s/it]
Epoch Progress:  44%|██████████▋             | 444/1000 [20:27<25:34,  2.76s/it]
Epoch Progress:  44%|██████████▋             | 445/1000 [20:30<25:31,  2.76s/it]
Epoch Progress:  45%|██████████▋             | 446/1000 [20:32<25:25,  2.75s/it]
Epoch Progress:  45%|██████████▋             | 447/1000 [20:35<25:22,  2.75s/it]
Epoch Progress:  45%|██████████▊             | 448/1000 [20:38<25:18,  2.75s/it]
Epoch Progress:  45%|██████████▊             | 449/1000 [20:41<25:17,  2.75s/it]
Epoch Progress:  45%|██████████▊             | 450/1000 [20:44<25:15,  2.76s/it]Dataset Size: 10%, Epoch: 451, Train Loss: 3.745646727712531, Val Loss: 11.949733147254356

Epoch Progress:  45%|██████████▊             | 451/1000 [20:46<25:12,  2.75s/it]
Epoch Progress:  45%|██████████▊             | 452/1000 [20:49<25:08,  2.75s/it]
Epoch Progress:  45%|██████████▊             | 453/1000 [20:52<25:03,  2.75s/it]
Epoch Progress:  45%|██████████▉             | 454/1000 [20:54<25:01,  2.75s/it]
Epoch Progress:  46%|██████████▉             | 455/1000 [20:57<24:58,  2.75s/it]
Epoch Progress:  46%|██████████▉             | 456/1000 [21:00<24:55,  2.75s/it]
Epoch Progress:  46%|██████████▉             | 457/1000 [21:03<24:53,  2.75s/it]
Epoch Progress:  46%|██████████▉             | 458/1000 [21:05<24:48,  2.75s/it]
Epoch Progress:  46%|███████████             | 459/1000 [21:08<24:45,  2.75s/it]
Epoch Progress:  46%|███████████             | 460/1000 [21:11<24:43,  2.75s/it]Dataset Size: 10%, Epoch: 461, Train Loss: 3.737155450017829, Val Loss: 12.00403179266514

Epoch Progress:  46%|███████████             | 461/1000 [21:14<24:41,  2.75s/it]
Epoch Progress:  46%|███████████             | 462/1000 [21:16<24:37,  2.75s/it]
Epoch Progress:  46%|███████████             | 463/1000 [21:19<24:35,  2.75s/it]
Epoch Progress:  46%|███████████▏            | 464/1000 [21:22<24:33,  2.75s/it]
Epoch Progress:  46%|███████████▏            | 465/1000 [21:25<24:34,  2.76s/it]
Epoch Progress:  47%|███████████▏            | 466/1000 [21:27<24:30,  2.75s/it]
Epoch Progress:  47%|███████████▏            | 467/1000 [21:30<24:27,  2.75s/it]
Epoch Progress:  47%|███████████▏            | 468/1000 [21:33<24:24,  2.75s/it]
Epoch Progress:  47%|███████████▎            | 469/1000 [21:36<24:21,  2.75s/it]
Epoch Progress:  47%|███████████▎            | 470/1000 [21:38<24:18,  2.75s/it]Dataset Size: 10%, Epoch: 471, Train Loss: 3.733100784452338, Val Loss: 12.089392735407902

Epoch Progress:  47%|███████████▎            | 471/1000 [21:41<24:17,  2.76s/it]
Epoch Progress:  47%|███████████▎            | 472/1000 [21:44<24:14,  2.76s/it]
Epoch Progress:  47%|███████████▎            | 473/1000 [21:47<24:10,  2.75s/it]
Epoch Progress:  47%|███████████▍            | 474/1000 [21:50<24:07,  2.75s/it]
Epoch Progress:  48%|███████████▍            | 475/1000 [21:52<24:05,  2.75s/it]
Epoch Progress:  48%|███████████▍            | 476/1000 [21:55<24:01,  2.75s/it]
Epoch Progress:  48%|███████████▍            | 477/1000 [21:58<23:59,  2.75s/it]
Epoch Progress:  48%|███████████▍            | 478/1000 [22:01<23:57,  2.75s/it]
Epoch Progress:  48%|███████████▍            | 479/1000 [22:03<23:53,  2.75s/it]
Epoch Progress:  48%|███████████▌            | 480/1000 [22:06<23:50,  2.75s/it]Dataset Size: 10%, Epoch: 481, Train Loss: 3.728038091408579, Val Loss: 12.176140418419472

Epoch Progress:  48%|███████████▌            | 481/1000 [22:09<23:48,  2.75s/it]
Epoch Progress:  48%|███████████▌            | 482/1000 [22:12<23:47,  2.76s/it]
Epoch Progress:  48%|███████████▌            | 483/1000 [22:14<23:40,  2.75s/it]
Epoch Progress:  48%|███████████▌            | 484/1000 [22:17<23:39,  2.75s/it]
Epoch Progress:  48%|███████████▋            | 485/1000 [22:20<23:37,  2.75s/it]
Epoch Progress:  49%|███████████▋            | 486/1000 [22:23<23:33,  2.75s/it]
Epoch Progress:  49%|███████████▋            | 487/1000 [22:25<23:29,  2.75s/it]
Epoch Progress:  49%|███████████▋            | 488/1000 [22:28<23:29,  2.75s/it]
Epoch Progress:  49%|███████████▋            | 489/1000 [22:31<23:29,  2.76s/it]
Epoch Progress:  49%|███████████▊            | 490/1000 [22:34<23:24,  2.75s/it]Dataset Size: 10%, Epoch: 491, Train Loss: 3.7194861236371493, Val Loss: 12.23568547077668

Epoch Progress:  49%|███████████▊            | 491/1000 [22:36<23:22,  2.76s/it]
Epoch Progress:  49%|███████████▊            | 492/1000 [22:39<23:20,  2.76s/it]
Epoch Progress:  49%|███████████▊            | 493/1000 [22:42<23:20,  2.76s/it]
Epoch Progress:  49%|███████████▊            | 494/1000 [22:45<23:13,  2.75s/it]
Epoch Progress:  50%|███████████▉            | 495/1000 [22:47<23:11,  2.75s/it]
Epoch Progress:  50%|███████████▉            | 496/1000 [22:50<23:08,  2.75s/it]
Epoch Progress:  50%|███████████▉            | 497/1000 [22:53<23:05,  2.76s/it]
Epoch Progress:  50%|███████████▉            | 498/1000 [22:56<23:02,  2.75s/it]
Epoch Progress:  50%|███████████▉            | 499/1000 [22:58<23:00,  2.76s/it]
Epoch Progress:  50%|████████████            | 500/1000 [23:01<23:01,  2.76s/it]Dataset Size: 10%, Epoch: 501, Train Loss: 3.7042103692104944, Val Loss: 12.312405928587301

Epoch Progress:  50%|████████████            | 501/1000 [23:04<22:59,  2.76s/it]
Epoch Progress:  50%|████████████            | 502/1000 [23:07<22:55,  2.76s/it]
Epoch Progress:  50%|████████████            | 503/1000 [23:09<22:52,  2.76s/it]
Epoch Progress:  50%|████████████            | 504/1000 [23:12<22:47,  2.76s/it]
Epoch Progress:  50%|████████████            | 505/1000 [23:15<22:46,  2.76s/it]
Epoch Progress:  51%|████████████▏           | 506/1000 [23:18<22:43,  2.76s/it]
Epoch Progress:  51%|████████████▏           | 507/1000 [23:20<22:40,  2.76s/it]
Epoch Progress:  51%|████████████▏           | 508/1000 [23:23<22:37,  2.76s/it]
Epoch Progress:  51%|████████████▏           | 509/1000 [23:26<22:34,  2.76s/it]
Epoch Progress:  51%|████████████▏           | 510/1000 [23:29<22:31,  2.76s/it]Dataset Size: 10%, Epoch: 511, Train Loss: 3.7154287350805184, Val Loss: 12.388019659580328

Epoch Progress:  51%|████████████▎           | 511/1000 [23:32<22:34,  2.77s/it]
Epoch Progress:  51%|████████████▎           | 512/1000 [23:34<22:28,  2.76s/it]
Epoch Progress:  51%|████████████▎           | 513/1000 [23:37<22:24,  2.76s/it]
Epoch Progress:  51%|████████████▎           | 514/1000 [23:40<22:20,  2.76s/it]
Epoch Progress:  52%|████████████▎           | 515/1000 [23:43<22:18,  2.76s/it]
Epoch Progress:  52%|████████████▍           | 516/1000 [23:45<22:13,  2.76s/it]
Epoch Progress:  52%|████████████▍           | 517/1000 [23:48<22:09,  2.75s/it]
Epoch Progress:  52%|████████████▍           | 518/1000 [23:51<22:08,  2.76s/it]
Epoch Progress:  52%|████████████▍           | 519/1000 [23:54<22:05,  2.76s/it]
Epoch Progress:  52%|████████████▍           | 520/1000 [23:56<22:02,  2.75s/it]Dataset Size: 10%, Epoch: 521, Train Loss: 3.7104846176348234, Val Loss: 12.43302313486735

Epoch Progress:  52%|████████████▌           | 521/1000 [23:59<21:59,  2.75s/it]
Epoch Progress:  52%|████████████▌           | 522/1000 [24:02<21:56,  2.75s/it]
Epoch Progress:  52%|████████████▌           | 523/1000 [24:05<21:53,  2.75s/it]
Epoch Progress:  52%|████████████▌           | 524/1000 [24:07<21:50,  2.75s/it]
Epoch Progress:  52%|████████████▌           | 525/1000 [24:10<21:46,  2.75s/it]
Epoch Progress:  53%|████████████▌           | 526/1000 [24:13<21:45,  2.75s/it]
Epoch Progress:  53%|████████████▋           | 527/1000 [24:16<21:42,  2.75s/it]
Epoch Progress:  53%|████████████▋           | 528/1000 [24:18<21:38,  2.75s/it]
Epoch Progress:  53%|████████████▋           | 529/1000 [24:21<21:37,  2.76s/it]
Epoch Progress:  53%|████████████▋           | 530/1000 [24:24<21:33,  2.75s/it]Dataset Size: 10%, Epoch: 531, Train Loss: 3.6990576857014705, Val Loss: 12.459811430711012

Epoch Progress:  53%|████████████▋           | 531/1000 [24:27<21:30,  2.75s/it]
Epoch Progress:  53%|████████████▊           | 532/1000 [24:29<21:27,  2.75s/it]
Epoch Progress:  53%|████████████▊           | 533/1000 [24:32<21:24,  2.75s/it]
Epoch Progress:  53%|████████████▊           | 534/1000 [24:35<21:25,  2.76s/it]
Epoch Progress:  54%|████████████▊           | 535/1000 [24:38<21:20,  2.75s/it]
Epoch Progress:  54%|████████████▊           | 536/1000 [24:40<21:14,  2.75s/it]
Epoch Progress:  54%|████████████▉           | 537/1000 [24:43<21:09,  2.74s/it]
Epoch Progress:  54%|████████████▉           | 538/1000 [24:46<21:05,  2.74s/it]
Epoch Progress:  54%|████████████▉           | 539/1000 [24:49<21:00,  2.74s/it]
Epoch Progress:  54%|████████████▉           | 540/1000 [24:51<20:58,  2.74s/it]Dataset Size: 10%, Epoch: 541, Train Loss: 3.696329060353731, Val Loss: 12.56759210733267

Epoch Progress:  54%|████████████▉           | 541/1000 [24:54<20:55,  2.73s/it]
Epoch Progress:  54%|█████████████           | 542/1000 [24:57<20:51,  2.73s/it]
Epoch Progress:  54%|█████████████           | 543/1000 [24:59<20:47,  2.73s/it]
Epoch Progress:  54%|█████████████           | 544/1000 [25:02<20:47,  2.74s/it]
Epoch Progress:  55%|█████████████           | 545/1000 [25:05<20:43,  2.73s/it]
Epoch Progress:  55%|█████████████           | 546/1000 [25:08<20:42,  2.74s/it]
Epoch Progress:  55%|█████████████▏          | 547/1000 [25:10<20:38,  2.73s/it]
Epoch Progress:  55%|█████████████▏          | 548/1000 [25:13<20:35,  2.73s/it]
Epoch Progress:  55%|█████████████▏          | 549/1000 [25:16<20:33,  2.73s/it]
Epoch Progress:  55%|█████████████▏          | 550/1000 [25:19<20:30,  2.73s/it]Dataset Size: 10%, Epoch: 551, Train Loss: 3.6760124093607853, Val Loss: 12.601715992658566

Epoch Progress:  55%|█████████████▏          | 551/1000 [25:21<20:28,  2.74s/it]
Epoch Progress:  55%|█████████████▏          | 552/1000 [25:24<20:26,  2.74s/it]
Epoch Progress:  55%|█████████████▎          | 553/1000 [25:27<20:22,  2.73s/it]
Epoch Progress:  55%|█████████████▎          | 554/1000 [25:30<20:19,  2.73s/it]
Epoch Progress:  56%|█████████████▎          | 555/1000 [25:32<20:16,  2.73s/it]
Epoch Progress:  56%|█████████████▎          | 556/1000 [25:35<20:14,  2.74s/it]
Epoch Progress:  56%|█████████████▎          | 557/1000 [25:38<20:13,  2.74s/it]
Epoch Progress:  56%|█████████████▍          | 558/1000 [25:40<20:09,  2.74s/it]
Epoch Progress:  56%|█████████████▍          | 559/1000 [25:43<20:06,  2.74s/it]
Epoch Progress:  56%|█████████████▍          | 560/1000 [25:46<20:02,  2.73s/it]Dataset Size: 10%, Epoch: 561, Train Loss: 3.688173036826284, Val Loss: 12.689913920867138

Epoch Progress:  56%|█████████████▍          | 561/1000 [25:49<19:59,  2.73s/it]
Epoch Progress:  56%|█████████████▍          | 562/1000 [25:51<19:57,  2.73s/it]
Epoch Progress:  56%|█████████████▌          | 563/1000 [25:54<19:55,  2.74s/it]
Epoch Progress:  56%|█████████████▌          | 564/1000 [25:57<19:53,  2.74s/it]
Epoch Progress:  56%|█████████████▌          | 565/1000 [26:00<19:50,  2.74s/it]
Epoch Progress:  57%|█████████████▌          | 566/1000 [26:02<19:46,  2.73s/it]
Epoch Progress:  57%|█████████████▌          | 567/1000 [26:05<19:41,  2.73s/it]
Epoch Progress:  57%|█████████████▋          | 568/1000 [26:08<19:36,  2.72s/it]
Epoch Progress:  57%|█████████████▋          | 569/1000 [26:11<19:34,  2.72s/it]
Epoch Progress:  57%|█████████████▋          | 570/1000 [26:13<19:34,  2.73s/it]Dataset Size: 10%, Epoch: 571, Train Loss: 3.680859603379902, Val Loss: 12.721915514041216

Epoch Progress:  57%|█████████████▋          | 571/1000 [26:16<19:32,  2.73s/it]
Epoch Progress:  57%|█████████████▋          | 572/1000 [26:19<19:31,  2.74s/it]
Epoch Progress:  57%|█████████████▊          | 573/1000 [26:22<19:30,  2.74s/it]
Epoch Progress:  57%|█████████████▊          | 574/1000 [26:24<19:27,  2.74s/it]
Epoch Progress:  57%|█████████████▊          | 575/1000 [26:27<19:23,  2.74s/it]
Epoch Progress:  58%|█████████████▊          | 576/1000 [26:30<19:21,  2.74s/it]
Epoch Progress:  58%|█████████████▊          | 577/1000 [26:32<19:19,  2.74s/it]
Epoch Progress:  58%|█████████████▊          | 578/1000 [26:35<19:17,  2.74s/it]
Epoch Progress:  58%|█████████████▉          | 579/1000 [26:38<19:15,  2.74s/it]
Epoch Progress:  58%|█████████████▉          | 580/1000 [26:41<19:12,  2.74s/it]Dataset Size: 10%, Epoch: 581, Train Loss: 3.6794197873065344, Val Loss: 12.78758797278771

Epoch Progress:  58%|█████████████▉          | 581/1000 [26:43<19:11,  2.75s/it]
Epoch Progress:  58%|█████████████▉          | 582/1000 [26:46<19:07,  2.75s/it]
Epoch Progress:  58%|█████████████▉          | 583/1000 [26:49<19:04,  2.74s/it]
Epoch Progress:  58%|██████████████          | 584/1000 [26:52<19:02,  2.75s/it]
Epoch Progress:  58%|██████████████          | 585/1000 [26:54<18:59,  2.75s/it]
Epoch Progress:  59%|██████████████          | 586/1000 [26:57<18:56,  2.75s/it]
Epoch Progress:  59%|██████████████          | 587/1000 [27:00<18:53,  2.74s/it]
Epoch Progress:  59%|██████████████          | 588/1000 [27:03<18:50,  2.74s/it]
Epoch Progress:  59%|██████████████▏         | 589/1000 [27:05<18:48,  2.75s/it]
Epoch Progress:  59%|██████████████▏         | 590/1000 [27:08<18:43,  2.74s/it]Dataset Size: 10%, Epoch: 591, Train Loss: 3.6763124214975456, Val Loss: 12.834257737184183

Epoch Progress:  59%|██████████████▏         | 591/1000 [27:11<18:42,  2.74s/it]
Epoch Progress:  59%|██████████████▏         | 592/1000 [27:14<18:42,  2.75s/it]
Epoch Progress:  59%|██████████████▏         | 593/1000 [27:16<18:38,  2.75s/it]
Epoch Progress:  59%|██████████████▎         | 594/1000 [27:19<18:35,  2.75s/it]
Epoch Progress:  60%|██████████████▎         | 595/1000 [27:22<18:32,  2.75s/it]
Epoch Progress:  60%|██████████████▎         | 596/1000 [27:25<18:29,  2.75s/it]
Epoch Progress:  60%|██████████████▎         | 597/1000 [27:27<18:25,  2.74s/it]
Epoch Progress:  60%|██████████████▎         | 598/1000 [27:30<18:22,  2.74s/it]
Epoch Progress:  60%|██████████████▍         | 599/1000 [27:33<18:20,  2.74s/it]
Epoch Progress:  60%|██████████████▍         | 600/1000 [27:36<18:16,  2.74s/it]Dataset Size: 10%, Epoch: 601, Train Loss: 3.674813565454985, Val Loss: 12.853978768373148

Epoch Progress:  60%|██████████████▍         | 601/1000 [27:38<18:21,  2.76s/it]
Epoch Progress:  60%|██████████████▍         | 602/1000 [27:41<18:18,  2.76s/it]
Epoch Progress:  60%|██████████████▍         | 603/1000 [27:44<18:13,  2.75s/it]
Epoch Progress:  60%|██████████████▍         | 604/1000 [27:47<18:11,  2.76s/it]
Epoch Progress:  60%|██████████████▌         | 605/1000 [27:49<18:09,  2.76s/it]
Epoch Progress:  61%|██████████████▌         | 606/1000 [27:52<18:06,  2.76s/it]
Epoch Progress:  61%|██████████████▌         | 607/1000 [27:55<18:02,  2.76s/it]
Epoch Progress:  61%|██████████████▌         | 608/1000 [27:58<17:58,  2.75s/it]
Epoch Progress:  61%|██████████████▌         | 609/1000 [28:00<17:55,  2.75s/it]
Epoch Progress:  61%|██████████████▋         | 610/1000 [28:03<17:51,  2.75s/it]Dataset Size: 10%, Epoch: 611, Train Loss: 3.6662396004325464, Val Loss: 12.932095625461676

Epoch Progress:  61%|██████████████▋         | 611/1000 [28:06<17:49,  2.75s/it]
Epoch Progress:  61%|██████████████▋         | 612/1000 [28:09<17:45,  2.75s/it]
Epoch Progress:  61%|██████████████▋         | 613/1000 [28:11<17:43,  2.75s/it]
Epoch Progress:  61%|██████████████▋         | 614/1000 [28:14<17:38,  2.74s/it]
Epoch Progress:  62%|██████████████▊         | 615/1000 [28:17<17:38,  2.75s/it]
Epoch Progress:  62%|██████████████▊         | 616/1000 [28:20<17:36,  2.75s/it]
Epoch Progress:  62%|██████████████▊         | 617/1000 [28:22<17:33,  2.75s/it]
Epoch Progress:  62%|██████████████▊         | 618/1000 [28:25<17:30,  2.75s/it]
Epoch Progress:  62%|██████████████▊         | 619/1000 [28:28<17:28,  2.75s/it]
Epoch Progress:  62%|██████████████▉         | 620/1000 [28:31<17:27,  2.76s/it]Dataset Size: 10%, Epoch: 621, Train Loss: 3.6621894710942318, Val Loss: 12.985907774705153

Epoch Progress:  62%|██████████████▉         | 621/1000 [28:33<17:21,  2.75s/it]
Epoch Progress:  62%|██████████████▉         | 622/1000 [28:36<17:18,  2.75s/it]
Epoch Progress:  62%|██████████████▉         | 623/1000 [28:39<17:15,  2.75s/it]
Epoch Progress:  62%|██████████████▉         | 624/1000 [28:42<17:14,  2.75s/it]
Epoch Progress:  62%|███████████████         | 625/1000 [28:44<17:11,  2.75s/it]
Epoch Progress:  63%|███████████████         | 626/1000 [28:47<17:08,  2.75s/it]
Epoch Progress:  63%|███████████████         | 627/1000 [28:50<17:06,  2.75s/it]
Epoch Progress:  63%|███████████████         | 628/1000 [28:53<17:03,  2.75s/it]
Epoch Progress:  63%|███████████████         | 629/1000 [28:55<17:00,  2.75s/it]
Epoch Progress:  63%|███████████████         | 630/1000 [28:58<16:58,  2.75s/it]Dataset Size: 10%, Epoch: 631, Train Loss: 3.663519501686096, Val Loss: 13.031889768747183

Epoch Progress:  63%|███████████████▏        | 631/1000 [29:01<16:59,  2.76s/it]
Epoch Progress:  63%|███████████████▏        | 632/1000 [29:04<16:56,  2.76s/it]
Epoch Progress:  63%|███████████████▏        | 633/1000 [29:06<16:55,  2.77s/it]
Epoch Progress:  63%|███████████████▏        | 634/1000 [29:09<16:52,  2.77s/it]
Epoch Progress:  64%|███████████████▏        | 635/1000 [29:12<16:50,  2.77s/it]
Epoch Progress:  64%|███████████████▎        | 636/1000 [29:15<16:46,  2.77s/it]
Epoch Progress:  64%|███████████████▎        | 637/1000 [29:18<16:43,  2.77s/it]
Epoch Progress:  64%|███████████████▎        | 638/1000 [29:20<16:43,  2.77s/it]
Epoch Progress:  64%|███████████████▎        | 639/1000 [29:23<16:40,  2.77s/it]
Epoch Progress:  64%|███████████████▎        | 640/1000 [29:26<16:34,  2.76s/it]Dataset Size: 10%, Epoch: 641, Train Loss: 3.65912159493095, Val Loss: 13.061064671247433

Epoch Progress:  64%|███████████████▍        | 641/1000 [29:29<16:30,  2.76s/it]
Epoch Progress:  64%|███████████████▍        | 642/1000 [29:31<16:29,  2.76s/it]
Epoch Progress:  64%|███████████████▍        | 643/1000 [29:34<16:25,  2.76s/it]
Epoch Progress:  64%|███████████████▍        | 644/1000 [29:37<16:21,  2.76s/it]
Epoch Progress:  64%|███████████████▍        | 645/1000 [29:40<16:17,  2.75s/it]
Epoch Progress:  65%|███████████████▌        | 646/1000 [29:42<16:14,  2.75s/it]
Epoch Progress:  65%|███████████████▌        | 647/1000 [29:45<16:11,  2.75s/it]
Epoch Progress:  65%|███████████████▌        | 648/1000 [29:48<16:05,  2.74s/it]
Epoch Progress:  65%|███████████████▌        | 649/1000 [29:51<16:02,  2.74s/it]
Epoch Progress:  65%|███████████████▌        | 650/1000 [29:53<16:01,  2.75s/it]Dataset Size: 10%, Epoch: 651, Train Loss: 3.656880598319204, Val Loss: 13.091436826265776

Epoch Progress:  65%|███████████████▌        | 651/1000 [29:56<15:57,  2.74s/it]
Epoch Progress:  65%|███████████████▋        | 652/1000 [29:59<15:53,  2.74s/it]
Epoch Progress:  65%|███████████████▋        | 653/1000 [30:02<15:52,  2.74s/it]
Epoch Progress:  65%|███████████████▋        | 654/1000 [30:04<15:49,  2.75s/it]
Epoch Progress:  66%|███████████████▋        | 655/1000 [30:07<15:46,  2.74s/it]
Epoch Progress:  66%|███████████████▋        | 656/1000 [30:10<15:43,  2.74s/it]
Epoch Progress:  66%|███████████████▊        | 657/1000 [30:13<15:41,  2.74s/it]
Epoch Progress:  66%|███████████████▊        | 658/1000 [30:15<15:38,  2.74s/it]
Epoch Progress:  66%|███████████████▊        | 659/1000 [30:18<15:33,  2.74s/it]
Epoch Progress:  66%|███████████████▊        | 660/1000 [30:21<15:32,  2.74s/it]Dataset Size: 10%, Epoch: 661, Train Loss: 3.6437853449269344, Val Loss: 13.138120553432367

Epoch Progress:  66%|███████████████▊        | 661/1000 [30:24<15:31,  2.75s/it]
Epoch Progress:  66%|███████████████▉        | 662/1000 [30:26<15:28,  2.75s/it]
Epoch Progress:  66%|███████████████▉        | 663/1000 [30:29<15:24,  2.74s/it]
Epoch Progress:  66%|███████████████▉        | 664/1000 [30:32<15:24,  2.75s/it]
Epoch Progress:  66%|███████████████▉        | 665/1000 [30:35<15:22,  2.75s/it]
Epoch Progress:  67%|███████████████▉        | 666/1000 [30:37<15:19,  2.75s/it]
Epoch Progress:  67%|████████████████        | 667/1000 [30:40<15:15,  2.75s/it]
Epoch Progress:  67%|████████████████        | 668/1000 [30:43<15:12,  2.75s/it]
Epoch Progress:  67%|████████████████        | 669/1000 [30:46<15:08,  2.75s/it]
Epoch Progress:  67%|████████████████        | 670/1000 [30:48<15:04,  2.74s/it]Dataset Size: 10%, Epoch: 671, Train Loss: 3.6433736587825574, Val Loss: 13.168587293380346

Epoch Progress:  67%|████████████████        | 671/1000 [30:51<15:02,  2.74s/it]
Epoch Progress:  67%|████████████████▏       | 672/1000 [30:54<14:59,  2.74s/it]
Epoch Progress:  67%|████████████████▏       | 673/1000 [30:56<14:56,  2.74s/it]
Epoch Progress:  67%|████████████████▏       | 674/1000 [30:59<14:54,  2.74s/it]
Epoch Progress:  68%|████████████████▏       | 675/1000 [31:02<14:51,  2.74s/it]
Epoch Progress:  68%|████████████████▏       | 676/1000 [31:05<14:49,  2.75s/it]
Epoch Progress:  68%|████████████████▏       | 677/1000 [31:07<14:48,  2.75s/it]
Epoch Progress:  68%|████████████████▎       | 678/1000 [31:10<14:44,  2.75s/it]
Epoch Progress:  68%|████████████████▎       | 679/1000 [31:13<14:41,  2.75s/it]
Epoch Progress:  68%|████████████████▎       | 680/1000 [31:16<14:37,  2.74s/it]Dataset Size: 10%, Epoch: 681, Train Loss: 3.6459036751797327, Val Loss: 13.209047488677196

Epoch Progress:  68%|████████████████▎       | 681/1000 [31:18<14:35,  2.75s/it]
Epoch Progress:  68%|████████████████▎       | 682/1000 [31:21<14:35,  2.75s/it]
Epoch Progress:  68%|████████████████▍       | 683/1000 [31:24<14:31,  2.75s/it]
Epoch Progress:  68%|████████████████▍       | 684/1000 [31:27<14:29,  2.75s/it]
Epoch Progress:  68%|████████████████▍       | 685/1000 [31:29<14:26,  2.75s/it]
Epoch Progress:  69%|████████████████▍       | 686/1000 [31:32<14:23,  2.75s/it]
Epoch Progress:  69%|████████████████▍       | 687/1000 [31:35<14:19,  2.75s/it]
Epoch Progress:  69%|████████████████▌       | 688/1000 [31:38<14:16,  2.75s/it]
Epoch Progress:  69%|████████████████▌       | 689/1000 [31:40<14:14,  2.75s/it]
Epoch Progress:  69%|████████████████▌       | 690/1000 [31:43<14:11,  2.75s/it]Dataset Size: 10%, Epoch: 691, Train Loss: 3.6449802925712182, Val Loss: 13.276008361425156

Epoch Progress:  69%|████████████████▌       | 691/1000 [31:46<14:07,  2.74s/it]
Epoch Progress:  69%|████████████████▌       | 692/1000 [31:49<14:04,  2.74s/it]
Epoch Progress:  69%|████████████████▋       | 693/1000 [31:51<14:02,  2.74s/it]
Epoch Progress:  69%|████████████████▋       | 694/1000 [31:54<13:59,  2.74s/it]
Epoch Progress:  70%|████████████████▋       | 695/1000 [31:57<13:56,  2.74s/it]
Epoch Progress:  70%|████████████████▋       | 696/1000 [32:00<13:54,  2.75s/it]
Epoch Progress:  70%|████████████████▋       | 697/1000 [32:02<13:52,  2.75s/it]
Epoch Progress:  70%|████████████████▊       | 698/1000 [32:05<13:48,  2.74s/it]
Epoch Progress:  70%|████████████████▊       | 699/1000 [32:08<13:45,  2.74s/it]
Epoch Progress:  70%|████████████████▊       | 700/1000 [32:11<13:43,  2.75s/it]Dataset Size: 10%, Epoch: 701, Train Loss: 3.635000379461991, Val Loss: 13.278338627937513

Epoch Progress:  70%|████████████████▊       | 701/1000 [32:13<13:42,  2.75s/it]
Epoch Progress:  70%|████████████████▊       | 702/1000 [32:16<13:39,  2.75s/it]
Epoch Progress:  70%|████████████████▊       | 703/1000 [32:19<13:37,  2.75s/it]
Epoch Progress:  70%|████████████████▉       | 704/1000 [32:22<13:34,  2.75s/it]
Epoch Progress:  70%|████████████████▉       | 705/1000 [32:24<13:31,  2.75s/it]
Epoch Progress:  71%|████████████████▉       | 706/1000 [32:27<13:29,  2.75s/it]
Epoch Progress:  71%|████████████████▉       | 707/1000 [32:30<13:27,  2.75s/it]
Epoch Progress:  71%|████████████████▉       | 708/1000 [32:33<13:24,  2.75s/it]
Epoch Progress:  71%|█████████████████       | 709/1000 [32:35<13:20,  2.75s/it]
Epoch Progress:  71%|█████████████████       | 710/1000 [32:38<13:16,  2.75s/it]Dataset Size: 10%, Epoch: 711, Train Loss: 3.639594322756717, Val Loss: 13.311353438939804

Epoch Progress:  71%|█████████████████       | 711/1000 [32:41<13:14,  2.75s/it]
Epoch Progress:  71%|█████████████████       | 712/1000 [32:44<13:11,  2.75s/it]
Epoch Progress:  71%|█████████████████       | 713/1000 [32:46<13:07,  2.74s/it]
Epoch Progress:  71%|█████████████████▏      | 714/1000 [32:49<13:04,  2.74s/it]
Epoch Progress:  72%|█████████████████▏      | 715/1000 [32:52<13:02,  2.74s/it]
Epoch Progress:  72%|█████████████████▏      | 716/1000 [32:55<12:59,  2.74s/it]
Epoch Progress:  72%|█████████████████▏      | 717/1000 [32:57<12:56,  2.74s/it]
Epoch Progress:  72%|█████████████████▏      | 718/1000 [33:00<12:53,  2.74s/it]
Epoch Progress:  72%|█████████████████▎      | 719/1000 [33:03<12:51,  2.75s/it]
Epoch Progress:  72%|█████████████████▎      | 720/1000 [33:06<12:48,  2.74s/it]Dataset Size: 10%, Epoch: 721, Train Loss: 3.636212010132639, Val Loss: 13.328654533777481

Epoch Progress:  72%|█████████████████▎      | 721/1000 [33:08<12:44,  2.74s/it]
Epoch Progress:  72%|█████████████████▎      | 722/1000 [33:11<12:43,  2.75s/it]
Epoch Progress:  72%|█████████████████▎      | 723/1000 [33:14<12:40,  2.74s/it]
Epoch Progress:  72%|█████████████████▍      | 724/1000 [33:17<12:36,  2.74s/it]
Epoch Progress:  72%|█████████████████▍      | 725/1000 [33:19<12:34,  2.74s/it]
Epoch Progress:  73%|█████████████████▍      | 726/1000 [33:22<12:31,  2.74s/it]
Epoch Progress:  73%|█████████████████▍      | 727/1000 [33:25<12:28,  2.74s/it]
Epoch Progress:  73%|█████████████████▍      | 728/1000 [33:28<12:26,  2.74s/it]
Epoch Progress:  73%|█████████████████▍      | 729/1000 [33:30<12:23,  2.74s/it]
Epoch Progress:  73%|█████████████████▌      | 730/1000 [33:33<12:22,  2.75s/it]Dataset Size: 10%, Epoch: 731, Train Loss: 3.6265390421214856, Val Loss: 13.376538447844677

Epoch Progress:  73%|█████████████████▌      | 731/1000 [33:36<12:19,  2.75s/it]
Epoch Progress:  73%|█████████████████▌      | 732/1000 [33:39<12:16,  2.75s/it]
Epoch Progress:  73%|█████████████████▌      | 733/1000 [33:41<12:13,  2.75s/it]
Epoch Progress:  73%|█████████████████▌      | 734/1000 [33:44<12:10,  2.74s/it]
Epoch Progress:  74%|█████████████████▋      | 735/1000 [33:47<12:07,  2.74s/it]
Epoch Progress:  74%|█████████████████▋      | 736/1000 [33:50<12:04,  2.75s/it]
Epoch Progress:  74%|█████████████████▋      | 737/1000 [33:52<12:01,  2.74s/it]
Epoch Progress:  74%|█████████████████▋      | 738/1000 [33:55<11:58,  2.74s/it]
Epoch Progress:  74%|█████████████████▋      | 739/1000 [33:58<11:55,  2.74s/it]
Epoch Progress:  74%|█████████████████▊      | 740/1000 [34:00<11:52,  2.74s/it]Dataset Size: 10%, Epoch: 741, Train Loss: 3.626938085806997, Val Loss: 13.408614134177183

Epoch Progress:  74%|█████████████████▊      | 741/1000 [34:03<11:50,  2.74s/it]
Epoch Progress:  74%|█████████████████▊      | 742/1000 [34:06<11:48,  2.75s/it]
Epoch Progress:  74%|█████████████████▊      | 743/1000 [34:09<11:45,  2.74s/it]
Epoch Progress:  74%|█████████████████▊      | 744/1000 [34:11<11:42,  2.74s/it]
Epoch Progress:  74%|█████████████████▉      | 745/1000 [34:14<11:39,  2.74s/it]
Epoch Progress:  75%|█████████████████▉      | 746/1000 [34:17<11:36,  2.74s/it]
Epoch Progress:  75%|█████████████████▉      | 747/1000 [34:20<11:33,  2.74s/it]
Epoch Progress:  75%|█████████████████▉      | 748/1000 [34:22<11:31,  2.74s/it]
Epoch Progress:  75%|█████████████████▉      | 749/1000 [34:25<11:28,  2.74s/it]
Epoch Progress:  75%|██████████████████      | 750/1000 [34:28<11:26,  2.74s/it]Dataset Size: 10%, Epoch: 751, Train Loss: 3.6306136783800627, Val Loss: 13.43645435724503

Epoch Progress:  75%|██████████████████      | 751/1000 [34:31<11:23,  2.75s/it]
Epoch Progress:  75%|██████████████████      | 752/1000 [34:33<11:20,  2.74s/it]
Epoch Progress:  75%|██████████████████      | 753/1000 [34:36<11:17,  2.74s/it]
Epoch Progress:  75%|██████████████████      | 754/1000 [34:39<11:15,  2.74s/it]
Epoch Progress:  76%|██████████████████      | 755/1000 [34:42<11:12,  2.75s/it]
Epoch Progress:  76%|██████████████████▏     | 756/1000 [34:44<11:09,  2.74s/it]
Epoch Progress:  76%|██████████████████▏     | 757/1000 [34:47<11:07,  2.75s/it]
Epoch Progress:  76%|██████████████████▏     | 758/1000 [34:50<11:04,  2.75s/it]
Epoch Progress:  76%|██████████████████▏     | 759/1000 [34:53<11:02,  2.75s/it]
Epoch Progress:  76%|██████████████████▏     | 760/1000 [34:55<10:59,  2.75s/it]Dataset Size: 10%, Epoch: 761, Train Loss: 3.627787345334103, Val Loss: 13.45630442790496

Epoch Progress:  76%|██████████████████▎     | 761/1000 [34:58<10:55,  2.74s/it]
Epoch Progress:  76%|██████████████████▎     | 762/1000 [35:01<10:53,  2.75s/it]
Epoch Progress:  76%|██████████████████▎     | 763/1000 [35:04<10:50,  2.74s/it]
Epoch Progress:  76%|██████████████████▎     | 764/1000 [35:06<10:47,  2.74s/it]
Epoch Progress:  76%|██████████████████▎     | 765/1000 [35:09<10:46,  2.75s/it]
Epoch Progress:  77%|██████████████████▍     | 766/1000 [35:12<10:43,  2.75s/it]
Epoch Progress:  77%|██████████████████▍     | 767/1000 [35:15<10:40,  2.75s/it]
Epoch Progress:  77%|██████████████████▍     | 768/1000 [35:17<10:37,  2.75s/it]
Epoch Progress:  77%|██████████████████▍     | 769/1000 [35:20<10:35,  2.75s/it]
Epoch Progress:  77%|██████████████████▍     | 770/1000 [35:23<10:33,  2.75s/it]Dataset Size: 10%, Epoch: 771, Train Loss: 3.6170301751086584, Val Loss: 13.492218188750439

Epoch Progress:  77%|██████████████████▌     | 771/1000 [35:26<10:29,  2.75s/it]
Epoch Progress:  77%|██████████████████▌     | 772/1000 [35:28<10:24,  2.74s/it]
Epoch Progress:  77%|██████████████████▌     | 773/1000 [35:31<10:21,  2.74s/it]
Epoch Progress:  77%|██████████████████▌     | 774/1000 [35:34<10:18,  2.74s/it]
Epoch Progress:  78%|██████████████████▌     | 775/1000 [35:37<10:14,  2.73s/it]
Epoch Progress:  78%|██████████████████▌     | 776/1000 [35:39<10:11,  2.73s/it]
Epoch Progress:  78%|██████████████████▋     | 777/1000 [35:42<10:10,  2.74s/it]
Epoch Progress:  78%|██████████████████▋     | 778/1000 [35:45<10:07,  2.74s/it]
Epoch Progress:  78%|██████████████████▋     | 779/1000 [35:47<10:03,  2.73s/it]
Epoch Progress:  78%|██████████████████▋     | 780/1000 [35:50<10:01,  2.73s/it]Dataset Size: 10%, Epoch: 781, Train Loss: 3.626554896956996, Val Loss: 13.4856747847337

Epoch Progress:  78%|██████████████████▋     | 781/1000 [35:53<09:58,  2.73s/it]
Epoch Progress:  78%|██████████████████▊     | 782/1000 [35:56<09:55,  2.73s/it]
Epoch Progress:  78%|██████████████████▊     | 783/1000 [35:58<09:52,  2.73s/it]
Epoch Progress:  78%|██████████████████▊     | 784/1000 [36:01<09:50,  2.73s/it]
Epoch Progress:  78%|██████████████████▊     | 785/1000 [36:04<09:47,  2.73s/it]
Epoch Progress:  79%|██████████████████▊     | 786/1000 [36:07<09:44,  2.73s/it]
Epoch Progress:  79%|██████████████████▉     | 787/1000 [36:09<09:41,  2.73s/it]
Epoch Progress:  79%|██████████████████▉     | 788/1000 [36:12<09:40,  2.74s/it]
Epoch Progress:  79%|██████████████████▉     | 789/1000 [36:15<09:36,  2.73s/it]
Epoch Progress:  79%|██████████████████▉     | 790/1000 [36:17<09:33,  2.73s/it]Dataset Size: 10%, Epoch: 791, Train Loss: 3.6105436965038904, Val Loss: 13.50286916586069

Epoch Progress:  79%|██████████████████▉     | 791/1000 [36:20<09:30,  2.73s/it]
Epoch Progress:  79%|███████████████████     | 792/1000 [36:23<09:28,  2.73s/it]
Epoch Progress:  79%|███████████████████     | 793/1000 [36:26<09:24,  2.73s/it]
Epoch Progress:  79%|███████████████████     | 794/1000 [36:28<09:21,  2.73s/it]
Epoch Progress:  80%|███████████████████     | 795/1000 [36:31<09:20,  2.74s/it]
Epoch Progress:  80%|███████████████████     | 796/1000 [36:34<09:18,  2.74s/it]
Epoch Progress:  80%|███████████████████▏    | 797/1000 [36:37<09:15,  2.74s/it]
Epoch Progress:  80%|███████████████████▏    | 798/1000 [36:39<09:13,  2.74s/it]
Epoch Progress:  80%|███████████████████▏    | 799/1000 [36:42<09:10,  2.74s/it]
Epoch Progress:  80%|███████████████████▏    | 800/1000 [36:45<09:08,  2.74s/it]Dataset Size: 10%, Epoch: 801, Train Loss: 3.611589971341585, Val Loss: 13.508343011904985

Epoch Progress:  80%|███████████████████▏    | 801/1000 [36:48<09:05,  2.74s/it]
Epoch Progress:  80%|███████████████████▏    | 802/1000 [36:50<09:03,  2.74s/it]
Epoch Progress:  80%|███████████████████▎    | 803/1000 [36:53<09:00,  2.74s/it]
Epoch Progress:  80%|███████████████████▎    | 804/1000 [36:56<08:57,  2.74s/it]
Epoch Progress:  80%|███████████████████▎    | 805/1000 [36:59<08:53,  2.74s/it]
Epoch Progress:  81%|███████████████████▎    | 806/1000 [37:01<08:52,  2.74s/it]
Epoch Progress:  81%|███████████████████▎    | 807/1000 [37:04<08:49,  2.74s/it]
Epoch Progress:  81%|███████████████████▍    | 808/1000 [37:07<08:45,  2.74s/it]
Epoch Progress:  81%|███████████████████▍    | 809/1000 [37:10<08:43,  2.74s/it]
Epoch Progress:  81%|███████████████████▍    | 810/1000 [37:12<08:40,  2.74s/it]Dataset Size: 10%, Epoch: 811, Train Loss: 3.61509907245636, Val Loss: 13.547446886698404

Epoch Progress:  81%|███████████████████▍    | 811/1000 [37:15<08:38,  2.74s/it]
Epoch Progress:  81%|███████████████████▍    | 812/1000 [37:18<08:35,  2.74s/it]
Epoch Progress:  81%|███████████████████▌    | 813/1000 [37:21<08:32,  2.74s/it]
Epoch Progress:  81%|███████████████████▌    | 814/1000 [37:23<08:29,  2.74s/it]
Epoch Progress:  82%|███████████████████▌    | 815/1000 [37:26<08:26,  2.74s/it]
Epoch Progress:  82%|███████████████████▌    | 816/1000 [37:29<08:23,  2.74s/it]
Epoch Progress:  82%|███████████████████▌    | 817/1000 [37:31<08:21,  2.74s/it]
Epoch Progress:  82%|███████████████████▋    | 818/1000 [37:34<08:18,  2.74s/it]
Epoch Progress:  82%|███████████████████▋    | 819/1000 [37:37<08:14,  2.73s/it]
Epoch Progress:  82%|███████████████████▋    | 820/1000 [37:40<08:12,  2.73s/it]Dataset Size: 10%, Epoch: 821, Train Loss: 3.6147176090039705, Val Loss: 13.534292759039463

Epoch Progress:  82%|███████████████████▋    | 821/1000 [37:42<08:10,  2.74s/it]
Epoch Progress:  82%|███████████████████▋    | 822/1000 [37:45<08:07,  2.74s/it]
Epoch Progress:  82%|███████████████████▊    | 823/1000 [37:48<08:04,  2.74s/it]
Epoch Progress:  82%|███████████████████▊    | 824/1000 [37:51<08:02,  2.74s/it]
Epoch Progress:  82%|███████████████████▊    | 825/1000 [37:53<07:59,  2.74s/it]
Epoch Progress:  83%|███████████████████▊    | 826/1000 [37:56<07:55,  2.73s/it]
Epoch Progress:  83%|███████████████████▊    | 827/1000 [37:59<07:51,  2.73s/it]
Epoch Progress:  83%|███████████████████▊    | 828/1000 [38:02<07:49,  2.73s/it]
Epoch Progress:  83%|███████████████████▉    | 829/1000 [38:04<07:45,  2.72s/it]
Epoch Progress:  83%|███████████████████▉    | 830/1000 [38:07<07:42,  2.72s/it]Dataset Size: 10%, Epoch: 831, Train Loss: 3.618439059508474, Val Loss: 13.553735439593975

Epoch Progress:  83%|███████████████████▉    | 831/1000 [38:10<07:39,  2.72s/it]
Epoch Progress:  83%|███████████████████▉    | 832/1000 [38:12<07:37,  2.72s/it]
Epoch Progress:  83%|███████████████████▉    | 833/1000 [38:15<07:34,  2.72s/it]
Epoch Progress:  83%|████████████████████    | 834/1000 [38:18<07:31,  2.72s/it]
Epoch Progress:  84%|████████████████████    | 835/1000 [38:21<07:29,  2.72s/it]
Epoch Progress:  84%|████████████████████    | 836/1000 [38:23<07:26,  2.72s/it]
Epoch Progress:  84%|████████████████████    | 837/1000 [38:26<07:23,  2.72s/it]
Epoch Progress:  84%|████████████████████    | 838/1000 [38:29<07:21,  2.72s/it]
Epoch Progress:  84%|████████████████████▏   | 839/1000 [38:31<07:18,  2.72s/it]
Epoch Progress:  84%|████████████████████▏   | 840/1000 [38:34<07:15,  2.72s/it]Dataset Size: 10%, Epoch: 841, Train Loss: 3.619635964694776, Val Loss: 13.572895661378519

Epoch Progress:  84%|████████████████████▏   | 841/1000 [38:37<07:12,  2.72s/it]
Epoch Progress:  84%|████████████████████▏   | 842/1000 [38:40<07:09,  2.72s/it]
Epoch Progress:  84%|████████████████████▏   | 843/1000 [38:42<07:07,  2.72s/it]
Epoch Progress:  84%|████████████████████▎   | 844/1000 [38:45<07:04,  2.72s/it]
Epoch Progress:  84%|████████████████████▎   | 845/1000 [38:48<07:01,  2.72s/it]
Epoch Progress:  85%|████████████████████▎   | 846/1000 [38:51<06:59,  2.73s/it]
Epoch Progress:  85%|████████████████████▎   | 847/1000 [38:53<06:56,  2.73s/it]
Epoch Progress:  85%|████████████████████▎   | 848/1000 [38:56<06:53,  2.72s/it]
Epoch Progress:  85%|████████████████████▍   | 849/1000 [38:59<06:51,  2.72s/it]
Epoch Progress:  85%|████████████████████▍   | 850/1000 [39:01<06:48,  2.72s/it]Dataset Size: 10%, Epoch: 851, Train Loss: 3.620078080578854, Val Loss: 13.59362680484087

Epoch Progress:  85%|████████████████████▍   | 851/1000 [39:04<06:45,  2.72s/it]
Epoch Progress:  85%|████████████████████▍   | 852/1000 [39:07<06:42,  2.72s/it]
Epoch Progress:  85%|████████████████████▍   | 853/1000 [39:10<06:39,  2.72s/it]
Epoch Progress:  85%|████████████████████▍   | 854/1000 [39:12<06:37,  2.72s/it]
Epoch Progress:  86%|████████████████████▌   | 855/1000 [39:15<06:34,  2.72s/it]
Epoch Progress:  86%|████████████████████▌   | 856/1000 [39:18<06:31,  2.72s/it]
Epoch Progress:  86%|████████████████████▌   | 857/1000 [39:20<06:28,  2.72s/it]
Epoch Progress:  86%|████████████████████▌   | 858/1000 [39:23<06:26,  2.72s/it]
Epoch Progress:  86%|████████████████████▌   | 859/1000 [39:26<06:23,  2.72s/it]
Epoch Progress:  86%|████████████████████▋   | 860/1000 [39:29<06:20,  2.72s/it]Dataset Size: 10%, Epoch: 861, Train Loss: 3.6163480909247148, Val Loss: 13.599108402545635

Epoch Progress:  86%|████████████████████▋   | 861/1000 [39:31<06:18,  2.72s/it]
Epoch Progress:  86%|████████████████████▋   | 862/1000 [39:34<06:15,  2.72s/it]
Epoch Progress:  86%|████████████████████▋   | 863/1000 [39:37<06:12,  2.72s/it]
Epoch Progress:  86%|████████████████████▋   | 864/1000 [39:39<06:10,  2.72s/it]
Epoch Progress:  86%|████████████████████▊   | 865/1000 [39:42<06:07,  2.72s/it]
Epoch Progress:  87%|████████████████████▊   | 866/1000 [39:45<06:04,  2.72s/it]
Epoch Progress:  87%|████████████████████▊   | 867/1000 [39:48<06:08,  2.77s/it]
Epoch Progress:  87%|████████████████████▊   | 868/1000 [39:51<06:03,  2.75s/it]
Epoch Progress:  87%|████████████████████▊   | 869/1000 [39:53<05:59,  2.74s/it]
Epoch Progress:  87%|████████████████████▉   | 870/1000 [39:56<05:56,  2.74s/it]Dataset Size: 10%, Epoch: 871, Train Loss: 3.612136571030868, Val Loss: 13.594747029818022

Epoch Progress:  87%|████████████████████▉   | 871/1000 [39:59<05:52,  2.73s/it]
Epoch Progress:  87%|████████████████████▉   | 872/1000 [40:01<05:49,  2.73s/it]
Epoch Progress:  87%|████████████████████▉   | 873/1000 [40:04<05:46,  2.73s/it]
Epoch Progress:  87%|████████████████████▉   | 874/1000 [40:07<05:43,  2.73s/it]
Epoch Progress:  88%|█████████████████████   | 875/1000 [40:10<05:40,  2.72s/it]
Epoch Progress:  88%|█████████████████████   | 876/1000 [40:12<05:37,  2.72s/it]
Epoch Progress:  88%|█████████████████████   | 877/1000 [40:15<05:34,  2.72s/it]
Epoch Progress:  88%|█████████████████████   | 878/1000 [40:18<05:31,  2.72s/it]
Epoch Progress:  88%|█████████████████████   | 879/1000 [40:20<05:28,  2.72s/it]
Epoch Progress:  88%|█████████████████████   | 880/1000 [40:23<05:26,  2.72s/it]Dataset Size: 10%, Epoch: 881, Train Loss: 3.608931397136889, Val Loss: 13.590880516247871

Epoch Progress:  88%|█████████████████████▏  | 881/1000 [40:26<05:24,  2.73s/it]
Epoch Progress:  88%|█████████████████████▏  | 882/1000 [40:29<05:21,  2.72s/it]
Epoch Progress:  88%|█████████████████████▏  | 883/1000 [40:31<05:18,  2.73s/it]
Epoch Progress:  88%|█████████████████████▏  | 884/1000 [40:34<05:15,  2.72s/it]
Epoch Progress:  88%|█████████████████████▏  | 885/1000 [40:37<05:13,  2.72s/it]
Epoch Progress:  89%|█████████████████████▎  | 886/1000 [40:40<05:10,  2.72s/it]
Epoch Progress:  89%|█████████████████████▎  | 887/1000 [40:42<05:07,  2.72s/it]
Epoch Progress:  89%|█████████████████████▎  | 888/1000 [40:45<05:04,  2.72s/it]
Epoch Progress:  89%|█████████████████████▎  | 889/1000 [40:48<05:02,  2.72s/it]
Epoch Progress:  89%|█████████████████████▎  | 890/1000 [40:50<04:59,  2.72s/it]Dataset Size: 10%, Epoch: 891, Train Loss: 3.614627329926742, Val Loss: 13.611401337843676

Epoch Progress:  89%|█████████████████████▍  | 891/1000 [40:53<04:56,  2.72s/it]
Epoch Progress:  89%|█████████████████████▍  | 892/1000 [40:56<04:54,  2.73s/it]
Epoch Progress:  89%|█████████████████████▍  | 893/1000 [40:59<04:52,  2.73s/it]
Epoch Progress:  89%|█████████████████████▍  | 894/1000 [41:01<04:49,  2.73s/it]
Epoch Progress:  90%|█████████████████████▍  | 895/1000 [41:04<04:46,  2.73s/it]
Epoch Progress:  90%|█████████████████████▌  | 896/1000 [41:07<04:43,  2.73s/it]
Epoch Progress:  90%|█████████████████████▌  | 897/1000 [41:09<04:40,  2.73s/it]
Epoch Progress:  90%|█████████████████████▌  | 898/1000 [41:12<04:37,  2.72s/it]
Epoch Progress:  90%|█████████████████████▌  | 899/1000 [41:15<04:35,  2.72s/it]
Epoch Progress:  90%|█████████████████████▌  | 900/1000 [41:18<04:32,  2.72s/it]Dataset Size: 10%, Epoch: 901, Train Loss: 3.609917163848877, Val Loss: 13.619748091086363

Epoch Progress:  90%|█████████████████████▌  | 901/1000 [41:20<04:29,  2.72s/it]
Epoch Progress:  90%|█████████████████████▋  | 902/1000 [41:23<04:26,  2.72s/it]
Epoch Progress:  90%|█████████████████████▋  | 903/1000 [41:26<04:23,  2.72s/it]
Epoch Progress:  90%|█████████████████████▋  | 904/1000 [41:29<04:21,  2.72s/it]
Epoch Progress:  90%|█████████████████████▋  | 905/1000 [41:31<04:19,  2.73s/it]
Epoch Progress:  91%|█████████████████████▋  | 906/1000 [41:34<04:16,  2.73s/it]
Epoch Progress:  91%|█████████████████████▊  | 907/1000 [41:37<04:13,  2.72s/it]
Epoch Progress:  91%|█████████████████████▊  | 908/1000 [41:39<04:10,  2.72s/it]
Epoch Progress:  91%|█████████████████████▊  | 909/1000 [41:42<04:07,  2.72s/it]
Epoch Progress:  91%|█████████████████████▊  | 910/1000 [41:45<04:04,  2.72s/it]Dataset Size: 10%, Epoch: 911, Train Loss: 3.610665428011041, Val Loss: 13.620887071658403

Epoch Progress:  91%|█████████████████████▊  | 911/1000 [41:48<04:02,  2.72s/it]
Epoch Progress:  91%|█████████████████████▉  | 912/1000 [41:50<03:59,  2.72s/it]
Epoch Progress:  91%|█████████████████████▉  | 913/1000 [41:53<03:56,  2.72s/it]
Epoch Progress:  91%|█████████████████████▉  | 914/1000 [41:56<03:53,  2.72s/it]
Epoch Progress:  92%|█████████████████████▉  | 915/1000 [41:58<03:51,  2.72s/it]
Epoch Progress:  92%|█████████████████████▉  | 916/1000 [42:01<03:49,  2.73s/it]
Epoch Progress:  92%|██████████████████████  | 917/1000 [42:04<03:46,  2.73s/it]
Epoch Progress:  92%|██████████████████████  | 918/1000 [42:07<03:43,  2.72s/it]
Epoch Progress:  92%|██████████████████████  | 919/1000 [42:09<03:40,  2.72s/it]
Epoch Progress:  92%|██████████████████████  | 920/1000 [42:12<03:37,  2.72s/it]Dataset Size: 10%, Epoch: 921, Train Loss: 3.613100578910426, Val Loss: 13.623153833242563

Epoch Progress:  92%|██████████████████████  | 921/1000 [42:15<03:34,  2.72s/it]
Epoch Progress:  92%|██████████████████████▏ | 922/1000 [42:18<03:32,  2.72s/it]
Epoch Progress:  92%|██████████████████████▏ | 923/1000 [42:20<03:29,  2.72s/it]
Epoch Progress:  92%|██████████████████████▏ | 924/1000 [42:23<03:26,  2.72s/it]
Epoch Progress:  92%|██████████████████████▏ | 925/1000 [42:26<03:23,  2.72s/it]
Epoch Progress:  93%|██████████████████████▏ | 926/1000 [42:28<03:21,  2.72s/it]
Epoch Progress:  93%|██████████████████████▏ | 927/1000 [42:31<03:18,  2.72s/it]
Epoch Progress:  93%|██████████████████████▎ | 928/1000 [42:34<03:16,  2.73s/it]
Epoch Progress:  93%|██████████████████████▎ | 929/1000 [42:37<03:13,  2.72s/it]
Epoch Progress:  93%|██████████████████████▎ | 930/1000 [42:39<03:10,  2.72s/it]Dataset Size: 10%, Epoch: 931, Train Loss: 3.6011771842053064, Val Loss: 13.637671641814404

Epoch Progress:  93%|██████████████████████▎ | 931/1000 [42:42<03:07,  2.72s/it]
Epoch Progress:  93%|██████████████████████▎ | 932/1000 [42:45<03:05,  2.72s/it]
Epoch Progress:  93%|██████████████████████▍ | 933/1000 [42:47<03:02,  2.72s/it]
Epoch Progress:  93%|██████████████████████▍ | 934/1000 [42:50<02:59,  2.72s/it]
Epoch Progress:  94%|██████████████████████▍ | 935/1000 [42:53<02:56,  2.72s/it]
Epoch Progress:  94%|██████████████████████▍ | 936/1000 [42:56<02:53,  2.72s/it]
Epoch Progress:  94%|██████████████████████▍ | 937/1000 [42:58<02:51,  2.72s/it]
Epoch Progress:  94%|██████████████████████▌ | 938/1000 [43:01<02:48,  2.72s/it]
Epoch Progress:  94%|██████████████████████▌ | 939/1000 [43:04<02:46,  2.72s/it]
Epoch Progress:  94%|██████████████████████▌ | 940/1000 [43:07<02:43,  2.72s/it]Dataset Size: 10%, Epoch: 941, Train Loss: 3.6065139394057426, Val Loss: 13.63292398208227

Epoch Progress:  94%|██████████████████████▌ | 941/1000 [43:09<02:40,  2.72s/it]
Epoch Progress:  94%|██████████████████████▌ | 942/1000 [43:12<02:38,  2.72s/it]
Epoch Progress:  94%|██████████████████████▋ | 943/1000 [43:15<02:35,  2.73s/it]
Epoch Progress:  94%|██████████████████████▋ | 944/1000 [43:17<02:32,  2.72s/it]
Epoch Progress:  94%|██████████████████████▋ | 945/1000 [43:20<02:29,  2.72s/it]
Epoch Progress:  95%|██████████████████████▋ | 946/1000 [43:23<02:27,  2.73s/it]
Epoch Progress:  95%|██████████████████████▋ | 947/1000 [43:26<02:24,  2.72s/it]
Epoch Progress:  95%|██████████████████████▊ | 948/1000 [43:28<02:21,  2.72s/it]
Epoch Progress:  95%|██████████████████████▊ | 949/1000 [43:31<02:18,  2.72s/it]
Epoch Progress:  95%|██████████████████████▊ | 950/1000 [43:34<02:16,  2.72s/it]Dataset Size: 10%, Epoch: 951, Train Loss: 3.6123349541111995, Val Loss: 13.641549672835913

Epoch Progress:  95%|██████████████████████▊ | 951/1000 [43:36<02:13,  2.72s/it]
Epoch Progress:  95%|██████████████████████▊ | 952/1000 [43:39<02:10,  2.72s/it]
Epoch Progress:  95%|██████████████████████▊ | 953/1000 [43:42<02:08,  2.72s/it]
Epoch Progress:  95%|██████████████████████▉ | 954/1000 [43:45<02:05,  2.72s/it]
Epoch Progress:  96%|██████████████████████▉ | 955/1000 [43:47<02:02,  2.72s/it]
Epoch Progress:  96%|██████████████████████▉ | 956/1000 [43:50<01:59,  2.72s/it]
Epoch Progress:  96%|██████████████████████▉ | 957/1000 [43:53<01:57,  2.72s/it]
Epoch Progress:  96%|██████████████████████▉ | 958/1000 [43:56<01:54,  2.72s/it]
Epoch Progress:  96%|███████████████████████ | 959/1000 [43:58<01:51,  2.72s/it]
Epoch Progress:  96%|███████████████████████ | 960/1000 [44:01<01:48,  2.72s/it]Dataset Size: 10%, Epoch: 961, Train Loss: 3.611554277570624, Val Loss: 13.634256216195913

Epoch Progress:  96%|███████████████████████ | 961/1000 [44:04<01:46,  2.72s/it]
Epoch Progress:  96%|███████████████████████ | 962/1000 [44:06<01:43,  2.72s/it]
Epoch Progress:  96%|███████████████████████ | 963/1000 [44:09<01:40,  2.72s/it]
Epoch Progress:  96%|███████████████████████▏| 964/1000 [44:12<01:38,  2.72s/it]
Epoch Progress:  96%|███████████████████████▏| 965/1000 [44:15<01:35,  2.72s/it]
Epoch Progress:  97%|███████████████████████▏| 966/1000 [44:17<01:32,  2.72s/it]
Epoch Progress:  97%|███████████████████████▏| 967/1000 [44:20<01:29,  2.72s/it]
Epoch Progress:  97%|███████████████████████▏| 968/1000 [44:23<01:27,  2.72s/it]
Epoch Progress:  97%|███████████████████████▎| 969/1000 [44:25<01:24,  2.72s/it]
Epoch Progress:  97%|███████████████████████▎| 970/1000 [44:28<01:21,  2.72s/it]Dataset Size: 10%, Epoch: 971, Train Loss: 3.6066189125964514, Val Loss: 13.638621452527168

Epoch Progress:  97%|███████████████████████▎| 971/1000 [44:31<01:18,  2.72s/it]
Epoch Progress:  97%|███████████████████████▎| 972/1000 [44:34<01:16,  2.72s/it]
Epoch Progress:  97%|███████████████████████▎| 973/1000 [44:36<01:13,  2.72s/it]
Epoch Progress:  97%|███████████████████████▍| 974/1000 [44:39<01:10,  2.72s/it]
Epoch Progress:  98%|███████████████████████▍| 975/1000 [44:42<01:08,  2.72s/it]
Epoch Progress:  98%|███████████████████████▍| 976/1000 [44:45<01:05,  2.72s/it]
Epoch Progress:  98%|███████████████████████▍| 977/1000 [44:47<01:02,  2.72s/it]
Epoch Progress:  98%|███████████████████████▍| 978/1000 [44:50<00:59,  2.72s/it]
Epoch Progress:  98%|███████████████████████▍| 979/1000 [44:53<00:57,  2.72s/it]
Epoch Progress:  98%|███████████████████████▌| 980/1000 [44:55<00:54,  2.72s/it]Dataset Size: 10%, Epoch: 981, Train Loss: 3.6101991753829155, Val Loss: 13.637197518960024

Epoch Progress:  98%|███████████████████████▌| 981/1000 [44:58<00:51,  2.72s/it]
Epoch Progress:  98%|███████████████████████▌| 982/1000 [45:01<00:49,  2.72s/it]
Epoch Progress:  98%|███████████████████████▌| 983/1000 [45:04<00:46,  2.72s/it]
Epoch Progress:  98%|███████████████████████▌| 984/1000 [45:06<00:43,  2.72s/it]
Epoch Progress:  98%|███████████████████████▋| 985/1000 [45:09<00:40,  2.72s/it]
Epoch Progress:  99%|███████████████████████▋| 986/1000 [45:12<00:38,  2.72s/it]
Epoch Progress:  99%|███████████████████████▋| 987/1000 [45:14<00:35,  2.72s/it]
Epoch Progress:  99%|███████████████████████▋| 988/1000 [45:17<00:32,  2.72s/it]
Epoch Progress:  99%|███████████████████████▋| 989/1000 [45:20<00:29,  2.72s/it]
Epoch Progress:  99%|███████████████████████▊| 990/1000 [45:23<00:27,  2.72s/it]Dataset Size: 10%, Epoch: 991, Train Loss: 3.6071200433530306, Val Loss: 13.63771445934589

Epoch Progress:  99%|███████████████████████▊| 991/1000 [45:25<00:24,  2.72s/it]
Epoch Progress:  99%|███████████████████████▊| 992/1000 [45:28<00:21,  2.72s/it]
Epoch Progress:  99%|███████████████████████▊| 993/1000 [45:31<00:19,  2.72s/it]
Epoch Progress:  99%|███████████████████████▊| 994/1000 [45:34<00:16,  2.72s/it]
Epoch Progress: 100%|███████████████████████▉| 995/1000 [45:36<00:13,  2.72s/it]
Epoch Progress: 100%|███████████████████████▉| 996/1000 [45:39<00:10,  2.72s/it]
Epoch Progress: 100%|███████████████████████▉| 997/1000 [45:42<00:08,  2.72s/it]
Epoch Progress: 100%|███████████████████████▉| 998/1000 [45:44<00:05,  2.73s/it]
Epoch Progress: 100%|███████████████████████▉| 999/1000 [45:47<00:02,  2.72s/it]
Epoch Progress: 100%|███████████████████████| 1000/1000 [45:50<00:00,  2.75s/it]
Dataset Size: 10%, Val Perplexity: 1302.8592529296875

Data Iteration: 100%|███████████████████████████| 1/1 [45:50<00:00, 2750.49s/it]
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
plt.title('Params=1.6M, Fraction=0.1, LR=0.001')
plt.legend()
plt.grid(True)
plt.savefig('1.6m_small_f_0.1_lr_0.001.png')
plt.show()