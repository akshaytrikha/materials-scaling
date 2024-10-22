import re
import pandas as pd

log_data = '''
Data Iteration:   0%|                                     | 0/1 [00:00<?, ?it/s]
Model is on device cuda and has 1622688 parameters

Epoch Progress:   0%|                                  | 0/1000 [00:00<?, ?it/s]Dataset Size: 50%, Epoch: 1, Train Loss: 10.978882892157442, Val Loss: 10.977441358566285

Epoch Progress:   0%|                        | 1/1000 [00:10<2:48:45, 10.14s/it]
Epoch Progress:   0%|                        | 2/1000 [00:19<2:44:59,  9.92s/it]
Epoch Progress:   0%|                        | 3/1000 [00:29<2:43:35,  9.85s/it]
Epoch Progress:   0%|                        | 4/1000 [00:39<2:42:50,  9.81s/it]
Epoch Progress:   0%|                        | 5/1000 [00:49<2:42:24,  9.79s/it]
Epoch Progress:   1%|▏                       | 6/1000 [00:58<2:42:01,  9.78s/it]
Epoch Progress:   1%|▏                       | 7/1000 [01:08<2:41:44,  9.77s/it]
Epoch Progress:   1%|▏                       | 8/1000 [01:18<2:41:27,  9.77s/it]
Epoch Progress:   1%|▏                       | 9/1000 [01:28<2:41:17,  9.77s/it]
Epoch Progress:   1%|▏                      | 10/1000 [01:37<2:41:03,  9.76s/it]Dataset Size: 50%, Epoch: 11, Train Loss: 9.793157157077584, Val Loss: 9.69707703590393

Epoch Progress:   1%|▎                      | 11/1000 [01:47<2:40:49,  9.76s/it]
Epoch Progress:   1%|▎                      | 12/1000 [01:57<2:40:42,  9.76s/it]
Epoch Progress:   1%|▎                      | 13/1000 [02:07<2:40:33,  9.76s/it]
Epoch Progress:   1%|▎                      | 14/1000 [02:16<2:40:19,  9.76s/it]
Epoch Progress:   2%|▎                      | 15/1000 [02:26<2:40:06,  9.75s/it]
Epoch Progress:   2%|▎                      | 16/1000 [02:36<2:39:58,  9.75s/it]
Epoch Progress:   2%|▍                      | 17/1000 [02:46<2:39:48,  9.75s/it]
Epoch Progress:   2%|▍                      | 18/1000 [02:55<2:39:39,  9.75s/it]
Epoch Progress:   2%|▍                      | 19/1000 [03:05<2:39:27,  9.75s/it]
Epoch Progress:   2%|▍                      | 20/1000 [03:15<2:39:14,  9.75s/it]Dataset Size: 50%, Epoch: 21, Train Loss: 7.538055066139467, Val Loss: 7.586632037162781

Epoch Progress:   2%|▍                      | 21/1000 [03:25<2:39:08,  9.75s/it]
Epoch Progress:   2%|▌                      | 22/1000 [03:35<2:38:59,  9.75s/it]
Epoch Progress:   2%|▌                      | 23/1000 [03:44<2:38:49,  9.75s/it]
Epoch Progress:   2%|▌                      | 24/1000 [03:54<2:38:37,  9.75s/it]
Epoch Progress:   2%|▌                      | 25/1000 [04:04<2:38:29,  9.75s/it]
Epoch Progress:   3%|▌                      | 26/1000 [04:14<2:38:22,  9.76s/it]
Epoch Progress:   3%|▌                      | 27/1000 [04:23<2:38:12,  9.76s/it]
Epoch Progress:   3%|▋                      | 28/1000 [04:33<2:38:01,  9.76s/it]
Epoch Progress:   3%|▋                      | 29/1000 [04:43<2:37:48,  9.75s/it]
Epoch Progress:   3%|▋                      | 30/1000 [04:53<2:37:36,  9.75s/it]Dataset Size: 50%, Epoch: 31, Train Loss: 6.952088299617972, Val Loss: 7.095497345924377

Epoch Progress:   3%|▋                      | 31/1000 [05:02<2:37:29,  9.75s/it]
Epoch Progress:   3%|▋                      | 32/1000 [05:12<2:37:22,  9.75s/it]
Epoch Progress:   3%|▊                      | 33/1000 [05:22<2:37:13,  9.76s/it]
Epoch Progress:   3%|▊                      | 34/1000 [05:32<2:37:05,  9.76s/it]
Epoch Progress:   4%|▊                      | 35/1000 [05:41<2:36:51,  9.75s/it]
Epoch Progress:   4%|▊                      | 36/1000 [05:51<2:36:41,  9.75s/it]
Epoch Progress:   4%|▊                      | 37/1000 [06:01<2:36:32,  9.75s/it]
Epoch Progress:   4%|▊                      | 38/1000 [06:11<2:36:26,  9.76s/it]
Epoch Progress:   4%|▉                      | 39/1000 [06:20<2:36:18,  9.76s/it]
Epoch Progress:   4%|▉                      | 40/1000 [06:30<2:36:07,  9.76s/it]Dataset Size: 50%, Epoch: 41, Train Loss: 6.4783134768086095, Val Loss: 6.786875796318054

Epoch Progress:   4%|▉                      | 41/1000 [06:40<2:35:59,  9.76s/it]
Epoch Progress:   4%|▉                      | 42/1000 [06:50<2:35:47,  9.76s/it]
Epoch Progress:   4%|▉                      | 43/1000 [06:59<2:35:36,  9.76s/it]
Epoch Progress:   4%|█                      | 44/1000 [07:09<2:35:29,  9.76s/it]
Epoch Progress:   4%|█                      | 45/1000 [07:19<2:35:17,  9.76s/it]
Epoch Progress:   5%|█                      | 46/1000 [07:29<2:35:06,  9.76s/it]
Epoch Progress:   5%|█                      | 47/1000 [07:38<2:35:02,  9.76s/it]
Epoch Progress:   5%|█                      | 48/1000 [07:48<2:34:57,  9.77s/it]
Epoch Progress:   5%|█▏                     | 49/1000 [07:58<2:34:43,  9.76s/it]
Epoch Progress:   5%|█▏                     | 50/1000 [08:08<2:34:38,  9.77s/it]Dataset Size: 50%, Epoch: 51, Train Loss: 6.109860635572864, Val Loss: 6.611709523200989

Epoch Progress:   5%|█▏                     | 51/1000 [08:17<2:34:26,  9.76s/it]
Epoch Progress:   5%|█▏                     | 52/1000 [08:27<2:34:02,  9.75s/it]
Epoch Progress:   5%|█▏                     | 53/1000 [08:37<2:33:54,  9.75s/it]
Epoch Progress:   5%|█▏                     | 54/1000 [08:47<2:33:57,  9.77s/it]
Epoch Progress:   6%|█▎                     | 55/1000 [08:57<2:33:52,  9.77s/it]
Epoch Progress:   6%|█▎                     | 56/1000 [09:06<2:33:59,  9.79s/it]
Epoch Progress:   6%|█▎                     | 57/1000 [09:16<2:34:12,  9.81s/it]
Epoch Progress:   6%|█▎                     | 58/1000 [09:26<2:34:08,  9.82s/it]
Epoch Progress:   6%|█▎                     | 59/1000 [09:36<2:33:57,  9.82s/it]
Epoch Progress:   6%|█▍                     | 60/1000 [09:46<2:33:27,  9.80s/it]Dataset Size: 50%, Epoch: 61, Train Loss: 5.826330138790992, Val Loss: 6.550340056419373

Epoch Progress:   6%|█▍                     | 61/1000 [09:55<2:33:09,  9.79s/it]
Epoch Progress:   6%|█▍                     | 62/1000 [10:05<2:33:04,  9.79s/it]
Epoch Progress:   6%|█▍                     | 63/1000 [10:15<2:32:59,  9.80s/it]
Epoch Progress:   6%|█▍                     | 64/1000 [10:25<2:32:47,  9.79s/it]
Epoch Progress:   6%|█▍                     | 65/1000 [10:35<2:32:50,  9.81s/it]
Epoch Progress:   7%|█▌                     | 66/1000 [10:44<2:32:54,  9.82s/it]
Epoch Progress:   7%|█▌                     | 67/1000 [10:54<2:32:39,  9.82s/it]
Epoch Progress:   7%|█▌                     | 68/1000 [11:04<2:32:25,  9.81s/it]
Epoch Progress:   7%|█▌                     | 69/1000 [11:14<2:31:50,  9.79s/it]
Epoch Progress:   7%|█▌                     | 70/1000 [11:24<2:31:23,  9.77s/it]Dataset Size: 50%, Epoch: 71, Train Loss: 5.604955893690868, Val Loss: 6.542569494247436

Epoch Progress:   7%|█▋                     | 71/1000 [11:33<2:30:59,  9.75s/it]
Epoch Progress:   7%|█▋                     | 72/1000 [11:43<2:30:41,  9.74s/it]
Epoch Progress:   7%|█▋                     | 73/1000 [11:53<2:30:24,  9.74s/it]
Epoch Progress:   7%|█▋                     | 74/1000 [12:02<2:30:08,  9.73s/it]
Epoch Progress:   8%|█▋                     | 75/1000 [12:12<2:29:54,  9.72s/it]
Epoch Progress:   8%|█▋                     | 76/1000 [12:22<2:29:44,  9.72s/it]
Epoch Progress:   8%|█▊                     | 77/1000 [12:32<2:29:32,  9.72s/it]
Epoch Progress:   8%|█▊                     | 78/1000 [12:41<2:29:18,  9.72s/it]
Epoch Progress:   8%|█▊                     | 79/1000 [12:51<2:29:08,  9.72s/it]
Epoch Progress:   8%|█▊                     | 80/1000 [13:01<2:28:59,  9.72s/it]Dataset Size: 50%, Epoch: 81, Train Loss: 5.439714190780475, Val Loss: 6.568093323707581

Epoch Progress:   8%|█▊                     | 81/1000 [13:10<2:28:50,  9.72s/it]
Epoch Progress:   8%|█▉                     | 82/1000 [13:20<2:29:25,  9.77s/it]
Epoch Progress:   8%|█▉                     | 83/1000 [13:30<2:29:16,  9.77s/it]
Epoch Progress:   8%|█▉                     | 84/1000 [13:40<2:29:10,  9.77s/it]
Epoch Progress:   8%|█▉                     | 85/1000 [13:50<2:28:48,  9.76s/it]
Epoch Progress:   9%|█▉                     | 86/1000 [13:59<2:28:31,  9.75s/it]
Epoch Progress:   9%|██                     | 87/1000 [14:09<2:28:13,  9.74s/it]
Epoch Progress:   9%|██                     | 88/1000 [14:19<2:27:55,  9.73s/it]
Epoch Progress:   9%|██                     | 89/1000 [14:28<2:27:43,  9.73s/it]
Epoch Progress:   9%|██                     | 90/1000 [14:38<2:27:29,  9.73s/it]Dataset Size: 50%, Epoch: 91, Train Loss: 5.3212797975027435, Val Loss: 6.636140322685241

Epoch Progress:   9%|██                     | 91/1000 [14:48<2:27:24,  9.73s/it]
Epoch Progress:   9%|██                     | 92/1000 [14:58<2:27:14,  9.73s/it]
Epoch Progress:   9%|██▏                    | 93/1000 [15:07<2:27:02,  9.73s/it]
Epoch Progress:   9%|██▏                    | 94/1000 [15:17<2:26:52,  9.73s/it]
Epoch Progress:  10%|██▏                    | 95/1000 [15:27<2:26:46,  9.73s/it]
Epoch Progress:  10%|██▏                    | 96/1000 [15:37<2:26:44,  9.74s/it]
Epoch Progress:  10%|██▏                    | 97/1000 [15:46<2:26:31,  9.74s/it]
Epoch Progress:  10%|██▎                    | 98/1000 [15:56<2:26:26,  9.74s/it]
Epoch Progress:  10%|██▎                    | 99/1000 [16:06<2:26:26,  9.75s/it]
Epoch Progress:  10%|██▏                   | 100/1000 [16:16<2:26:37,  9.77s/it]Dataset Size: 50%, Epoch: 101, Train Loss: 5.2386571412445395, Val Loss: 6.703924083709717

Epoch Progress:  10%|██▏                   | 101/1000 [16:25<2:26:44,  9.79s/it]
Epoch Progress:  10%|██▏                   | 102/1000 [16:35<2:26:47,  9.81s/it]
Epoch Progress:  10%|██▎                   | 103/1000 [16:45<2:26:32,  9.80s/it]
Epoch Progress:  10%|██▎                   | 104/1000 [16:55<2:26:03,  9.78s/it]
Epoch Progress:  10%|██▎                   | 105/1000 [17:05<2:25:52,  9.78s/it]
Epoch Progress:  11%|██▎                   | 106/1000 [17:14<2:25:33,  9.77s/it]
Epoch Progress:  11%|██▎                   | 107/1000 [17:24<2:25:16,  9.76s/it]
Epoch Progress:  11%|██▍                   | 108/1000 [17:34<2:25:05,  9.76s/it]
Epoch Progress:  11%|██▍                   | 109/1000 [17:44<2:24:52,  9.76s/it]
Epoch Progress:  11%|██▍                   | 110/1000 [17:53<2:24:37,  9.75s/it]Dataset Size: 50%, Epoch: 111, Train Loss: 5.180100399960754, Val Loss: 6.7728125810623165

Epoch Progress:  11%|██▍                   | 111/1000 [18:03<2:24:28,  9.75s/it]
Epoch Progress:  11%|██▍                   | 112/1000 [18:13<2:24:20,  9.75s/it]
Epoch Progress:  11%|██▍                   | 113/1000 [18:23<2:24:07,  9.75s/it]
Epoch Progress:  11%|██▌                   | 114/1000 [18:32<2:23:56,  9.75s/it]
Epoch Progress:  12%|██▌                   | 115/1000 [18:42<2:23:45,  9.75s/it]
Epoch Progress:  12%|██▌                   | 116/1000 [18:52<2:23:38,  9.75s/it]
Epoch Progress:  12%|██▌                   | 117/1000 [19:02<2:23:42,  9.77s/it]
Epoch Progress:  12%|██▌                   | 118/1000 [19:11<2:23:48,  9.78s/it]
Epoch Progress:  12%|██▌                   | 119/1000 [19:21<2:23:40,  9.79s/it]
Epoch Progress:  12%|██▋                   | 120/1000 [19:31<2:23:29,  9.78s/it]Dataset Size: 50%, Epoch: 121, Train Loss: 5.135527323651058, Val Loss: 6.841085433959961

Epoch Progress:  12%|██▋                   | 121/1000 [19:41<2:23:20,  9.78s/it]
Epoch Progress:  12%|██▋                   | 122/1000 [19:51<2:23:09,  9.78s/it]
Epoch Progress:  12%|██▋                   | 123/1000 [20:00<2:23:00,  9.78s/it]
Epoch Progress:  12%|██▋                   | 124/1000 [20:10<2:22:49,  9.78s/it]
Epoch Progress:  12%|██▊                   | 125/1000 [20:20<2:22:51,  9.80s/it]
Epoch Progress:  13%|██▊                   | 126/1000 [20:30<2:23:06,  9.82s/it]
Epoch Progress:  13%|██▊                   | 127/1000 [20:40<2:22:51,  9.82s/it]
Epoch Progress:  13%|██▊                   | 128/1000 [20:49<2:22:29,  9.80s/it]
Epoch Progress:  13%|██▊                   | 129/1000 [20:59<2:21:59,  9.78s/it]
Epoch Progress:  13%|██▊                   | 130/1000 [21:09<2:21:41,  9.77s/it]Dataset Size: 50%, Epoch: 131, Train Loss: 5.101558111047232, Val Loss: 6.890262246131897

Epoch Progress:  13%|██▉                   | 131/1000 [21:19<2:21:36,  9.78s/it]
Epoch Progress:  13%|██▉                   | 132/1000 [21:28<2:21:17,  9.77s/it]
Epoch Progress:  13%|██▉                   | 133/1000 [21:38<2:20:59,  9.76s/it]
Epoch Progress:  13%|██▉                   | 134/1000 [21:48<2:20:49,  9.76s/it]
Epoch Progress:  14%|██▉                   | 135/1000 [21:58<2:20:45,  9.76s/it]
Epoch Progress:  14%|██▉                   | 136/1000 [22:08<2:20:42,  9.77s/it]
Epoch Progress:  14%|███                   | 137/1000 [22:17<2:20:33,  9.77s/it]
Epoch Progress:  14%|███                   | 138/1000 [22:27<2:20:22,  9.77s/it]
Epoch Progress:  14%|███                   | 139/1000 [22:37<2:19:56,  9.75s/it]
Epoch Progress:  14%|███                   | 140/1000 [22:47<2:19:37,  9.74s/it]Dataset Size: 50%, Epoch: 141, Train Loss: 5.07523416191019, Val Loss: 6.94994375705719

Epoch Progress:  14%|███                   | 141/1000 [22:56<2:19:20,  9.73s/it]
Epoch Progress:  14%|███                   | 142/1000 [23:06<2:19:03,  9.72s/it]
Epoch Progress:  14%|███▏                  | 143/1000 [23:16<2:18:50,  9.72s/it]
Epoch Progress:  14%|███▏                  | 144/1000 [23:25<2:18:44,  9.73s/it]
Epoch Progress:  14%|███▏                  | 145/1000 [23:35<2:18:31,  9.72s/it]
Epoch Progress:  15%|███▏                  | 146/1000 [23:45<2:18:18,  9.72s/it]
Epoch Progress:  15%|███▏                  | 147/1000 [23:55<2:18:10,  9.72s/it]
Epoch Progress:  15%|███▎                  | 148/1000 [24:04<2:18:00,  9.72s/it]
Epoch Progress:  15%|███▎                  | 149/1000 [24:14<2:17:47,  9.71s/it]
Epoch Progress:  15%|███▎                  | 150/1000 [24:24<2:17:37,  9.72s/it]Dataset Size: 50%, Epoch: 151, Train Loss: 5.0535260272282425, Val Loss: 6.988314914703369

Epoch Progress:  15%|███▎                  | 151/1000 [24:33<2:17:30,  9.72s/it]
Epoch Progress:  15%|███▎                  | 152/1000 [24:43<2:17:15,  9.71s/it]
Epoch Progress:  15%|███▎                  | 153/1000 [24:53<2:17:04,  9.71s/it]
Epoch Progress:  15%|███▍                  | 154/1000 [25:03<2:16:57,  9.71s/it]
Epoch Progress:  16%|███▍                  | 155/1000 [25:12<2:16:48,  9.71s/it]
Epoch Progress:  16%|███▍                  | 156/1000 [25:22<2:16:39,  9.72s/it]
Epoch Progress:  16%|███▍                  | 157/1000 [25:32<2:16:31,  9.72s/it]
Epoch Progress:  16%|███▍                  | 158/1000 [25:41<2:16:21,  9.72s/it]
Epoch Progress:  16%|███▍                  | 159/1000 [25:51<2:16:09,  9.71s/it]
Epoch Progress:  16%|███▌                  | 160/1000 [26:01<2:15:57,  9.71s/it]Dataset Size: 50%, Epoch: 161, Train Loss: 5.036287174429945, Val Loss: 7.0388710498809814

Epoch Progress:  16%|███▌                  | 161/1000 [26:11<2:15:49,  9.71s/it]
Epoch Progress:  16%|███▌                  | 162/1000 [26:20<2:15:38,  9.71s/it]
Epoch Progress:  16%|███▌                  | 163/1000 [26:30<2:15:27,  9.71s/it]
Epoch Progress:  16%|███▌                  | 164/1000 [26:40<2:15:20,  9.71s/it]
Epoch Progress:  16%|███▋                  | 165/1000 [26:49<2:15:11,  9.71s/it]
Epoch Progress:  17%|███▋                  | 166/1000 [26:59<2:15:00,  9.71s/it]
Epoch Progress:  17%|███▋                  | 167/1000 [27:09<2:14:50,  9.71s/it]
Epoch Progress:  17%|███▋                  | 168/1000 [27:19<2:14:40,  9.71s/it]
Epoch Progress:  17%|███▋                  | 169/1000 [27:28<2:14:30,  9.71s/it]
Epoch Progress:  17%|███▋                  | 170/1000 [27:38<2:14:24,  9.72s/it]Dataset Size: 50%, Epoch: 171, Train Loss: 5.020189551896946, Val Loss: 7.068554425239563

Epoch Progress:  17%|███▊                  | 171/1000 [27:48<2:14:10,  9.71s/it]
Epoch Progress:  17%|███▊                  | 172/1000 [27:57<2:14:00,  9.71s/it]
Epoch Progress:  17%|███▊                  | 173/1000 [28:07<2:13:58,  9.72s/it]
Epoch Progress:  17%|███▊                  | 174/1000 [28:17<2:13:46,  9.72s/it]
Epoch Progress:  18%|███▊                  | 175/1000 [28:27<2:13:39,  9.72s/it]
Epoch Progress:  18%|███▊                  | 176/1000 [28:36<2:13:26,  9.72s/it]
Epoch Progress:  18%|███▉                  | 177/1000 [28:46<2:13:16,  9.72s/it]
Epoch Progress:  18%|███▉                  | 178/1000 [28:56<2:13:04,  9.71s/it]
Epoch Progress:  18%|███▉                  | 179/1000 [29:05<2:12:54,  9.71s/it]
Epoch Progress:  18%|███▉                  | 180/1000 [29:15<2:12:44,  9.71s/it]Dataset Size: 50%, Epoch: 181, Train Loss: 5.006733781547957, Val Loss: 7.079857993125915

Epoch Progress:  18%|███▉                  | 181/1000 [29:25<2:12:34,  9.71s/it]
Epoch Progress:  18%|████                  | 182/1000 [29:35<2:12:27,  9.72s/it]
Epoch Progress:  18%|████                  | 183/1000 [29:44<2:12:19,  9.72s/it]
Epoch Progress:  18%|████                  | 184/1000 [29:54<2:12:06,  9.71s/it]
Epoch Progress:  18%|████                  | 185/1000 [30:04<2:11:56,  9.71s/it]
Epoch Progress:  19%|████                  | 186/1000 [30:13<2:11:49,  9.72s/it]
Epoch Progress:  19%|████                  | 187/1000 [30:23<2:11:37,  9.71s/it]
Epoch Progress:  20%|████▎                 | 197/1000 [32:00<2:10:02,  9.72s/it]
Epoch Progress:  20%|████▎                 | 198/1000 [32:10<2:09:53,  9.72s/it]
Epoch Progress:  20%|████▍                 | 199/1000 [32:20<2:09:47,  9.72s/it]
Epoch Progress:  20%|████▍                 | 200/1000 [32:29<2:09:35,  9.72s/it]Dataset Size: 50%, Epoch: 201, Train Loss: 4.986013914949151, Val Loss: 7.164769887924194

Epoch Progress:  20%|████▍                 | 201/1000 [32:39<2:09:24,  9.72s/it]
Epoch Progress:  20%|████▍                 | 202/1000 [32:49<2:09:15,  9.72s/it]
Epoch Progress:  20%|████▍                 | 203/1000 [32:59<2:09:06,  9.72s/it]
Epoch Progress:  20%|████▍                 | 204/1000 [33:08<2:08:54,  9.72s/it]
Epoch Progress:  20%|████▌                 | 205/1000 [33:18<2:08:46,  9.72s/it]
Epoch Progress:  21%|████▌                 | 206/1000 [33:28<2:08:42,  9.73s/it]
Epoch Progress:  21%|████▌                 | 207/1000 [33:37<2:08:33,  9.73s/it]
Epoch Progress:  21%|████▌                 | 208/1000 [33:47<2:08:26,  9.73s/it]
Epoch Progress:  21%|████▌                 | 209/1000 [33:57<2:08:21,  9.74s/it]
Epoch Progress:  21%|████▌                 | 210/1000 [34:07<2:08:07,  9.73s/it]Dataset Size: 50%, Epoch: 211, Train Loss: 4.971341845809772, Val Loss: 7.188668298721313

Epoch Progress:  21%|████▋                 | 211/1000 [34:16<2:07:53,  9.73s/it]
Epoch Progress:  21%|████▋                 | 212/1000 [34:26<2:07:48,  9.73s/it]
Epoch Progress:  21%|████▋                 | 213/1000 [34:36<2:07:36,  9.73s/it]
Epoch Progress:  21%|████▋                 | 214/1000 [34:46<2:07:25,  9.73s/it]
Epoch Progress:  22%|████▋                 | 215/1000 [34:55<2:07:20,  9.73s/it]
Epoch Progress:  22%|████▊                 | 216/1000 [35:05<2:07:11,  9.73s/it]
Epoch Progress:  22%|████▊                 | 217/1000 [35:15<2:07:00,  9.73s/it]
Epoch Progress:  22%|████▊                 | 218/1000 [35:25<2:06:57,  9.74s/it]
Epoch Progress:  22%|████▊                 | 219/1000 [35:34<2:06:50,  9.75s/it]
Epoch Progress:  22%|████▊                 | 220/1000 [35:44<2:06:40,  9.74s/it]Dataset Size: 50%, Epoch: 221, Train Loss: 4.964417437071441, Val Loss: 7.214304351806641

Epoch Progress:  22%|████▊                 | 221/1000 [35:54<2:06:29,  9.74s/it]
Epoch Progress:  22%|████▉                 | 222/1000 [36:04<2:06:27,  9.75s/it]
Epoch Progress:  22%|████▉                 | 223/1000 [36:13<2:06:15,  9.75s/it]
Epoch Progress:  22%|████▉                 | 224/1000 [36:23<2:06:13,  9.76s/it]
Epoch Progress:  22%|████▉                 | 225/1000 [36:33<2:06:06,  9.76s/it]
Epoch Progress:  23%|████▉                 | 226/1000 [36:43<2:05:49,  9.75s/it]
Epoch Progress:  23%|████▉                 | 227/1000 [36:52<2:05:45,  9.76s/it]
Epoch Progress:  23%|█████                 | 228/1000 [37:02<2:05:35,  9.76s/it]
Epoch Progress:  23%|█████                 | 229/1000 [37:12<2:05:11,  9.74s/it]
Epoch Progress:  23%|█████                 | 230/1000 [37:22<2:04:50,  9.73s/it]Dataset Size: 50%, Epoch: 231, Train Loss: 4.959445907223609, Val Loss: 7.2490334749221805

Epoch Progress:  23%|█████                 | 231/1000 [37:31<2:04:36,  9.72s/it]
Epoch Progress:  23%|█████                 | 232/1000 [37:41<2:04:26,  9.72s/it]
Epoch Progress:  23%|█████▏                | 233/1000 [37:51<2:04:13,  9.72s/it]
Epoch Progress:  23%|█████▏                | 234/1000 [38:00<2:04:03,  9.72s/it]
Epoch Progress:  24%|█████▏                | 235/1000 [38:10<2:03:56,  9.72s/it]
Epoch Progress:  24%|█████▏                | 236/1000 [38:20<2:03:43,  9.72s/it]
Epoch Progress:  24%|█████▏                | 237/1000 [38:30<2:03:31,  9.71s/it]
Epoch Progress:  24%|█████▏                | 238/1000 [38:39<2:03:25,  9.72s/it]
Epoch Progress:  24%|█████▎                | 239/1000 [38:49<2:03:20,  9.72s/it]
Epoch Progress:  24%|█████▎                | 240/1000 [38:59<2:03:10,  9.72s/it]Dataset Size: 50%, Epoch: 241, Train Loss: 4.951947345528551, Val Loss: 7.284210205078125

Epoch Progress:  24%|█████▎                | 241/1000 [39:08<2:03:00,  9.72s/it]
Epoch Progress:  24%|█████▎                | 242/1000 [39:18<2:02:47,  9.72s/it]
Epoch Progress:  24%|█████▎                | 243/1000 [39:28<2:02:34,  9.71s/it]
Epoch Progress:  25%|█████▌                | 254/1000 [41:15<2:01:01,  9.73s/it]
Epoch Progress:  26%|█████▌                | 255/1000 [41:25<2:00:44,  9.72s/it]
Epoch Progress:  26%|█████▋                | 256/1000 [41:34<2:00:34,  9.72s/it]
Epoch Progress:  26%|█████▋                | 257/1000 [41:44<2:00:28,  9.73s/it]
Epoch Progress:  26%|█████▋                | 258/1000 [41:54<2:00:19,  9.73s/it]
Epoch Progress:  26%|█████▋                | 259/1000 [42:04<2:00:17,  9.74s/it]
Epoch Progress:  26%|█████▋                | 260/1000 [42:13<2:00:08,  9.74s/it]Dataset Size: 50%, Epoch: 261, Train Loss: 4.938853325382356, Val Loss: 7.3295804262161255

Epoch Progress:  26%|█████▋                | 261/1000 [42:23<2:00:28,  9.78s/it]
Epoch Progress:  26%|█████▊                | 262/1000 [42:33<2:00:04,  9.76s/it]
Epoch Progress:  26%|█████▊                | 263/1000 [42:43<1:59:45,  9.75s/it]
Epoch Progress:  26%|█████▊                | 264/1000 [42:52<1:59:32,  9.74s/it]
Epoch Progress:  26%|█████▊                | 265/1000 [43:02<1:59:15,  9.74s/it]
Epoch Progress:  27%|█████▊                | 266/1000 [43:12<1:59:00,  9.73s/it]
Epoch Progress:  27%|█████▊                | 267/1000 [43:22<1:58:53,  9.73s/it]
Epoch Progress:  27%|█████▉                | 268/1000 [43:31<1:58:41,  9.73s/it]
Epoch Progress:  27%|█████▉                | 269/1000 [43:41<1:58:35,  9.73s/it]
Epoch Progress:  27%|█████▉                | 270/1000 [43:51<1:58:54,  9.77s/it]Dataset Size: 50%, Epoch: 271, Train Loss: 4.93167301916307, Val Loss: 7.366822052001953

Epoch Progress:  27%|█████▉                | 271/1000 [44:01<1:58:56,  9.79s/it]
Epoch Progress:  27%|█████▉                | 272/1000 [44:11<1:58:43,  9.79s/it]
Epoch Progress:  27%|██████                | 273/1000 [44:20<1:58:39,  9.79s/it]
Epoch Progress:  27%|██████                | 274/1000 [44:30<1:58:17,  9.78s/it]
Epoch Progress:  28%|██████                | 275/1000 [44:40<1:57:56,  9.76s/it]
Epoch Progress:  28%|██████▎               | 285/1000 [46:17<1:55:47,  9.72s/it]
Epoch Progress:  29%|██████▎               | 286/1000 [46:27<1:55:40,  9.72s/it]
Epoch Progress:  29%|██████▎               | 287/1000 [46:36<1:55:34,  9.73s/it]
Epoch Progress:  29%|██████▎               | 288/1000 [46:46<1:55:19,  9.72s/it]
Epoch Progress:  29%|██████▎               | 289/1000 [46:56<1:55:09,  9.72s/it]
Epoch Progress:  29%|██████▍               | 290/1000 [47:06<1:55:03,  9.72s/it]Dataset Size: 50%, Epoch: 291, Train Loss: 4.918608562920683, Val Loss: 7.386947822570801

Epoch Progress:  29%|██████▍               | 291/1000 [47:15<1:54:50,  9.72s/it]
Epoch Progress:  29%|██████▍               | 292/1000 [47:25<1:54:38,  9.72s/it]
Epoch Progress:  29%|██████▍               | 293/1000 [47:35<1:54:32,  9.72s/it]
Epoch Progress:  29%|██████▍               | 294/1000 [47:44<1:54:23,  9.72s/it]
Epoch Progress:  30%|██████▍               | 295/1000 [47:54<1:54:10,  9.72s/it]
Epoch Progress:  30%|██████▌               | 296/1000 [48:04<1:54:05,  9.72s/it]
Epoch Progress:  30%|██████▌               | 297/1000 [48:14<1:53:54,  9.72s/it]
Epoch Progress:  30%|██████▌               | 298/1000 [48:23<1:53:43,  9.72s/it]
Epoch Progress:  30%|██████▌               | 299/1000 [48:33<1:53:32,  9.72s/it]
Epoch Progress:  30%|██████▌               | 300/1000 [48:43<1:53:24,  9.72s/it]Dataset Size: 50%, Epoch: 301, Train Loss: 4.9149808832394175, Val Loss: 7.405603504180908

Epoch Progress:  30%|██████▌               | 301/1000 [48:53<1:53:11,  9.72s/it]
Epoch Progress:  30%|██████▋               | 302/1000 [49:02<1:53:00,  9.71s/it]
Epoch Progress:  30%|██████▋               | 303/1000 [49:12<1:53:00,  9.73s/it]
Epoch Progress:  30%|██████▋               | 304/1000 [49:22<1:52:49,  9.73s/it]
Epoch Progress:  30%|██████▋               | 305/1000 [49:31<1:52:35,  9.72s/it]
Epoch Progress:  31%|██████▋               | 306/1000 [49:41<1:52:27,  9.72s/it]
Epoch Progress:  31%|██████▊               | 307/1000 [49:51<1:52:15,  9.72s/it]
Epoch Progress:  31%|██████▊               | 308/1000 [50:01<1:52:05,  9.72s/it]
Epoch Progress:  31%|██████▊               | 309/1000 [50:10<1:51:58,  9.72s/it]
Epoch Progress:  31%|██████▊               | 310/1000 [50:20<1:51:45,  9.72s/it]Dataset Size: 50%, Epoch: 311, Train Loss: 4.908411174692134, Val Loss: 7.420509624481201

Epoch Progress:  31%|██████▊               | 311/1000 [50:30<1:51:34,  9.72s/it]
Epoch Progress:  31%|██████▊               | 312/1000 [50:39<1:51:25,  9.72s/it]
Epoch Progress:  31%|██████▉               | 313/1000 [50:49<1:51:18,  9.72s/it]
Epoch Progress:  31%|██████▉               | 314/1000 [50:59<1:51:08,  9.72s/it]
Epoch Progress:  32%|██████▉               | 315/1000 [51:09<1:50:56,  9.72s/it]
Epoch Progress:  32%|██████▉               | 316/1000 [51:18<1:50:50,  9.72s/it]
Epoch Progress:  32%|██████▉               | 317/1000 [51:28<1:50:38,  9.72s/it]
Epoch Progress:  32%|██████▉               | 318/1000 [51:38<1:50:24,  9.71s/it]
Epoch Progress:  32%|███████               | 319/1000 [51:47<1:50:18,  9.72s/it]
Epoch Progress:  32%|███████               | 320/1000 [51:57<1:50:07,  9.72s/it]Dataset Size: 50%, Epoch: 321, Train Loss: 4.90864586061047, Val Loss: 7.438485860824585

Epoch Progress:  32%|███████               | 321/1000 [52:07<1:49:58,  9.72s/it]
Epoch Progress:  32%|███████               | 322/1000 [52:17<1:49:51,  9.72s/it]
Epoch Progress:  32%|███████               | 323/1000 [52:26<1:49:39,  9.72s/it]
Epoch Progress:  32%|███████▏              | 324/1000 [52:36<1:49:32,  9.72s/it]
Epoch Progress:  32%|███████▏              | 325/1000 [52:46<1:49:30,  9.73s/it]
Epoch Progress:  33%|███████▏              | 326/1000 [52:56<1:49:22,  9.74s/it]
Epoch Progress:  33%|███████▏              | 327/1000 [53:05<1:49:06,  9.73s/it]
Epoch Progress:  33%|███████▏              | 328/1000 [53:15<1:48:57,  9.73s/it]
Epoch Progress:  33%|███████▏              | 329/1000 [53:25<1:48:54,  9.74s/it]
Epoch Progress:  33%|███████▎              | 330/1000 [53:35<1:48:43,  9.74s/it]Dataset Size: 50%, Epoch: 331, Train Loss: 4.9048999765867825, Val Loss: 7.4694836139678955

Epoch Progress:  33%|███████▎              | 331/1000 [53:44<1:48:33,  9.74s/it]
Epoch Progress:  33%|███████▎              | 332/1000 [53:54<1:48:29,  9.74s/it]
Epoch Progress:  33%|███████▎              | 333/1000 [54:04<1:48:16,  9.74s/it]
Epoch Progress:  33%|███████▎              | 334/1000 [54:13<1:48:04,  9.74s/it]
Epoch Progress:  34%|███████▎              | 335/1000 [54:23<1:47:57,  9.74s/it]
Epoch Progress:  34%|███████▍              | 336/1000 [54:33<1:47:46,  9.74s/it]
Epoch Progress:  34%|███████▍              | 337/1000 [54:43<1:47:33,  9.73s/it]
Epoch Progress:  34%|███████▍              | 338/1000 [54:52<1:47:24,  9.74s/it]
Epoch Progress:  34%|███████▍              | 339/1000 [55:02<1:47:12,  9.73s/it]
Epoch Progress:  34%|███████▍              | 340/1000 [55:12<1:46:58,  9.73s/it]Dataset Size: 50%, Epoch: 341, Train Loss: 4.898054030633742, Val Loss: 7.479413890838623

Epoch Progress:  34%|███████▌              | 341/1000 [55:22<1:46:47,  9.72s/it]Dataset Size: 50%, Epoch: 351, Train Loss: 4.895769303844821, Val Loss: 7.492434000968933

Epoch Progress:  35%|███████▋              | 351/1000 [56:59<1:45:09,  9.72s/it]
Epoch Progress:  35%|███████▋              | 352/1000 [57:08<1:44:59,  9.72s/it]
Epoch Progress:  35%|███████▊              | 353/1000 [57:18<1:44:49,  9.72s/it]
Epoch Progress:  35%|███████▊              | 354/1000 [57:28<1:44:38,  9.72s/it]
Epoch Progress:  36%|███████▊              | 355/1000 [57:38<1:44:33,  9.73s/it]
Epoch Progress:  36%|███████▊              | 356/1000 [57:47<1:44:21,  9.72s/it]
Epoch Progress:  36%|███████▊              | 357/1000 [57:57<1:44:10,  9.72s/it]
Epoch Progress:  36%|███████▉              | 358/1000 [58:07<1:44:04,  9.73s/it]
Epoch Progress:  36%|███████▉              | 359/1000 [58:17<1:43:53,  9.72s/it]
Epoch Progress:  36%|███████▉              | 360/1000 [58:26<1:43:43,  9.72s/it]Dataset Size: 50%, Epoch: 361, Train Loss: 4.891526750338975, Val Loss: 7.521209311485291

Epoch Progress:  36%|███████▉              | 361/1000 [58:36<1:43:36,  9.73s/it]
Epoch Progress:  36%|███████▉              | 362/1000 [58:46<1:43:30,  9.73s/it]
Epoch Progress:  36%|███████▉              | 363/1000 [58:56<1:43:19,  9.73s/it]
Epoch Progress:  36%|████████              | 364/1000 [59:05<1:43:13,  9.74s/it]
Epoch Progress:  36%|████████              | 365/1000 [59:15<1:43:09,  9.75s/it]
Epoch Progress:  37%|████████              | 366/1000 [59:25<1:42:56,  9.74s/it]
Epoch Progress:  37%|████████              | 367/1000 [59:34<1:42:44,  9.74s/it]
Epoch Progress:  37%|████████              | 368/1000 [59:44<1:42:37,  9.74s/it]
Epoch Progress:  37%|████████              | 369/1000 [59:54<1:42:23,  9.74s/it]
Epoch Progress:  37%|███████▍            | 370/1000 [1:00:04<1:42:11,  9.73s/it]Dataset Size: 50%, Epoch: 371, Train Loss: 4.886659991356634, Val Loss: 7.540329456329346

Epoch Progress:  37%|███████▍            | 371/1000 [1:00:13<1:42:04,  9.74s/it]
Epoch Progress:  37%|███████▍            | 372/1000 [1:00:23<1:41:52,  9.73s/it]
Epoch Progress:  37%|███████▍            | 373/1000 [1:00:33<1:41:41,  9.73s/it]
Epoch Progress:  37%|███████▍            | 374/1000 [1:00:43<1:41:30,  9.73s/it]
Epoch Progress:  38%|███████▌            | 375/1000 [1:00:52<1:41:20,  9.73s/it]
Epoch Progress:  38%|███████▌            | 376/1000 [1:01:02<1:41:05,  9.72s/it]
Epoch Progress:  38%|███████▌            | 377/1000 [1:01:12<1:40:58,  9.72s/it]
Epoch Progress:  38%|███████▌            | 378/1000 [1:01:21<1:40:47,  9.72s/it]
Epoch Progress:  38%|███████▌            | 379/1000 [1:01:31<1:40:34,  9.72s/it]
Epoch Progress:  38%|███████▌            | 380/1000 [1:01:41<1:40:22,  9.71s/it]Dataset Size: 50%, Epoch: 381, Train Loss: 4.884444939192905, Val Loss: 7.548298740386963

Epoch Progress:  38%|███████▌            | 381/1000 [1:01:51<1:40:16,  9.72s/it]
Epoch Progress:  38%|███████▋            | 382/1000 [1:02:00<1:40:08,  9.72s/it]
Epoch Progress:  38%|███████▋            | 383/1000 [1:02:10<1:39:55,  9.72s/it]
Epoch Progress:  38%|███████▋            | 384/1000 [1:02:20<1:39:48,  9.72s/it]
Epoch Progress:  38%|███████▋            | 385/1000 [1:02:30<1:39:38,  9.72s/it]
Epoch Progress:  39%|███████▋            | 386/1000 [1:02:39<1:39:25,  9.72s/it]
Epoch Progress:  39%|███████▋            | 387/1000 [1:02:49<1:39:18,  9.72s/it]
Epoch Progress:  39%|███████▊            | 388/1000 [1:02:59<1:39:09,  9.72s/it]
Epoch Progress:  39%|███████▊            | 389/1000 [1:03:08<1:38:57,  9.72s/it]
Epoch Progress:  39%|███████▊            | 390/1000 [1:03:18<1:38:47,  9.72s/it]Dataset Size: 50%, Epoch: 391, Train Loss: 4.878074589595999, Val Loss: 7.607165789604187

Epoch Progress:  39%|███████▊            | 391/1000 [1:03:28<1:38:38,  9.72s/it]
Epoch Progress:  39%|███████▊            | 392/1000 [1:03:38<1:38:28,  9.72s/it]
Epoch Progress:  39%|███████▊            | 393/1000 [1:03:47<1:38:17,  9.72s/it]
Epoch Progress:  39%|███████▉            | 394/1000 [1:03:57<1:38:11,  9.72s/it]
Epoch Progress:  40%|███████▉            | 395/1000 [1:04:07<1:38:00,  9.72s/it]
Epoch Progress:  40%|███████▉            | 396/1000 [1:04:16<1:37:52,  9.72s/it]
Epoch Progress:  40%|███████▉            | 397/1000 [1:04:26<1:37:44,  9.73s/it]
Epoch Progress:  41%|████████▏           | 407/1000 [1:06:03<1:36:03,  9.72s/it]
Epoch Progress:  41%|████████▏           | 408/1000 [1:06:13<1:35:50,  9.71s/it]
Epoch Progress:  41%|████████▏           | 409/1000 [1:06:23<1:35:41,  9.72s/it]
Epoch Progress:  41%|████████▏           | 410/1000 [1:06:32<1:35:35,  9.72s/it]Dataset Size: 50%, Epoch: 411, Train Loss: 4.870009145429058, Val Loss: 7.617742276191711

Epoch Progress:  41%|████████▏           | 411/1000 [1:06:42<1:35:24,  9.72s/it]
Epoch Progress:  41%|████████▏           | 412/1000 [1:06:52<1:35:13,  9.72s/it]
Epoch Progress:  41%|████████▎           | 413/1000 [1:07:02<1:35:12,  9.73s/it]
Epoch Progress:  41%|████████▎           | 414/1000 [1:07:11<1:35:00,  9.73s/it]
Epoch Progress:  42%|████████▎           | 415/1000 [1:07:21<1:34:46,  9.72s/it]
Epoch Progress:  42%|████████▎           | 416/1000 [1:07:31<1:34:37,  9.72s/it]
Epoch Progress:  42%|████████▎           | 417/1000 [1:07:41<1:34:26,  9.72s/it]
Epoch Progress:  42%|████████▎           | 418/1000 [1:07:50<1:34:14,  9.72s/it]
Epoch Progress:  42%|████████▍           | 419/1000 [1:08:00<1:34:07,  9.72s/it]
Epoch Progress:  42%|████████▍           | 420/1000 [1:08:10<1:33:58,  9.72s/it]Dataset Size: 50%, Epoch: 421, Train Loss: 4.871245409852715, Val Loss: 7.640575814247131

Epoch Progress:  42%|████████▍           | 421/1000 [1:08:19<1:33:48,  9.72s/it]
Epoch Progress:  42%|████████▍           | 422/1000 [1:08:29<1:33:36,  9.72s/it]
Epoch Progress:  42%|████████▍           | 423/1000 [1:08:39<1:33:33,  9.73s/it]
Epoch Progress:  42%|████████▍           | 424/1000 [1:08:49<1:33:21,  9.72s/it]
Epoch Progress:  42%|████████▌           | 425/1000 [1:08:58<1:33:15,  9.73s/it]
Epoch Progress:  43%|████████▌           | 426/1000 [1:09:08<1:33:05,  9.73s/it]
Epoch Progress:  43%|████████▌           | 427/1000 [1:09:18<1:32:53,  9.73s/it]
Epoch Progress:  43%|████████▌           | 428/1000 [1:09:28<1:32:40,  9.72s/it]
Epoch Progress:  43%|████████▌           | 429/1000 [1:09:37<1:32:47,  9.75s/it]
Epoch Progress:  43%|████████▌           | 430/1000 [1:09:47<1:32:44,  9.76s/it]Dataset Size: 50%, Epoch: 431, Train Loss: 4.867776665636288, Val Loss: 7.629566025733948

Epoch Progress:  43%|████████▌           | 431/1000 [1:09:57<1:32:38,  9.77s/it]
Epoch Progress:  43%|████████▋           | 432/1000 [1:10:07<1:32:23,  9.76s/it]
Epoch Progress:  43%|████████▋           | 433/1000 [1:10:16<1:32:12,  9.76s/it]
Epoch Progress:  43%|████████▋           | 434/1000 [1:10:26<1:31:59,  9.75s/it]
Epoch Progress:  44%|████████▋           | 435/1000 [1:10:36<1:31:45,  9.74s/it]
Epoch Progress:  44%|████████▋           | 436/1000 [1:10:46<1:31:37,  9.75s/it]
Epoch Progress:  44%|████████▋           | 437/1000 [1:10:55<1:31:25,  9.74s/it]
Epoch Progress:  44%|████████▊           | 438/1000 [1:11:05<1:31:23,  9.76s/it]
Epoch Progress:  44%|████████▊           | 439/1000 [1:11:15<1:31:17,  9.76s/it]
Epoch Progress:  44%|████████▊           | 440/1000 [1:11:25<1:31:32,  9.81s/it]Dataset Size: 50%, Epoch: 441, Train Loss: 4.86186340291013, Val Loss: 7.667389917373657

Epoch Progress:  44%|████████▊           | 441/1000 [1:11:35<1:31:08,  9.78s/it]
Epoch Progress:  44%|████████▊           | 442/1000 [1:11:44<1:30:46,  9.76s/it]
Epoch Progress:  44%|████████▊           | 443/1000 [1:11:54<1:30:27,  9.74s/it]
Epoch Progress:  44%|████████▉           | 444/1000 [1:12:04<1:30:11,  9.73s/it]
Epoch Progress:  44%|████████▉           | 445/1000 [1:12:13<1:29:57,  9.73s/it]
Epoch Progress:  45%|████████▉           | 446/1000 [1:12:23<1:29:46,  9.72s/it]
Epoch Progress:  45%|████████▉           | 447/1000 [1:12:33<1:29:34,  9.72s/it]
Epoch Progress:  45%|████████▉           | 448/1000 [1:12:43<1:29:22,  9.71s/it]
Epoch Progress:  45%|████████▉           | 449/1000 [1:12:52<1:29:14,  9.72s/it]
Epoch Progress:  45%|█████████           | 450/1000 [1:13:02<1:29:04,  9.72s/it]Dataset Size: 50%, Epoch: 451, Train Loss: 4.857990095692296, Val Loss: 7.666389131546021

Epoch Progress:  45%|█████████           | 451/1000 [1:13:12<1:28:54,  9.72s/it]
Epoch Progress:  45%|█████████           | 452/1000 [1:13:21<1:28:44,  9.72s/it]
Epoch Progress:  45%|█████████           | 453/1000 [1:13:31<1:28:34,  9.72s/it]
Epoch Progress:  45%|█████████           | 454/1000 [1:13:41<1:28:23,  9.71s/it]
Epoch Progress:  46%|█████████           | 455/1000 [1:13:51<1:28:15,  9.72s/it]
Epoch Progress:  46%|█████████           | 456/1000 [1:14:00<1:28:06,  9.72s/it]
Epoch Progress:  46%|█████████▏          | 457/1000 [1:14:10<1:27:54,  9.71s/it]
Epoch Progress:  46%|█████████▏          | 458/1000 [1:14:20<1:27:45,  9.72s/it]
Epoch Progress:  46%|█████████▏          | 459/1000 [1:14:29<1:27:35,  9.72s/it]
Epoch Progress:  46%|█████████▏          | 460/1000 [1:14:39<1:27:26,  9.72s/it]Dataset Size: 50%, Epoch: 461, Train Loss: 4.8558909098307295, Val Loss: 7.6713563919067385

Epoch Progress:  46%|█████████▏          | 461/1000 [1:14:49<1:27:15,  9.71s/it]
Epoch Progress:  46%|█████████▏          | 462/1000 [1:14:59<1:27:08,  9.72s/it]
Epoch Progress:  46%|█████████▎          | 463/1000 [1:15:08<1:27:00,  9.72s/it]
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
plt.savefig('1.6m_small_f_0.5_lr_0.001.png')
plt.show()