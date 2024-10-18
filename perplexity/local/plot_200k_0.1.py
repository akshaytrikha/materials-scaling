import re
import pandas as pd

log_data = '''
Epoch Progress:   0%|                                  | 0/1000 [00:00<?, ?it/s]Dataset Size: 10%, Epoch: 1, Train Loss: 10.984898236592612, Val Loss: 10.981090607581201

Epoch Progress:   0%|                        | 1/1000 [00:03<1:03:56,  3.84s/it]
Epoch Progress:   0%|                          | 2/1000 [00:06<55:49,  3.36s/it]
Epoch Progress:   0%|                          | 3/1000 [00:09<53:11,  3.20s/it]
Epoch Progress:   0%|                          | 4/1000 [00:12<51:59,  3.13s/it]
Epoch Progress:   0%|▏                         | 5/1000 [00:15<51:15,  3.09s/it]
Epoch Progress:   1%|▏                         | 6/1000 [00:18<50:53,  3.07s/it]
Epoch Progress:   1%|▏                         | 7/1000 [00:21<50:39,  3.06s/it]
Epoch Progress:   1%|▏                         | 8/1000 [00:25<50:22,  3.05s/it]
Epoch Progress:   1%|▏                         | 9/1000 [00:28<50:09,  3.04s/it]
Epoch Progress:   1%|▎                        | 10/1000 [00:31<50:02,  3.03s/it]Dataset Size: 10%, Epoch: 11, Train Loss: 10.941180318196615, Val Loss: 10.940224536053545

Epoch Progress:   1%|▎                        | 11/1000 [00:34<50:01,  3.03s/it]
Epoch Progress:   1%|▎                        | 12/1000 [00:37<49:54,  3.03s/it]
Epoch Progress:   1%|▎                        | 13/1000 [00:40<49:46,  3.03s/it]
Epoch Progress:   1%|▎                        | 14/1000 [00:43<49:38,  3.02s/it]
Epoch Progress:   2%|▍                        | 15/1000 [00:46<49:38,  3.02s/it]
Epoch Progress:   2%|▍                        | 16/1000 [00:49<49:30,  3.02s/it]
Epoch Progress:   2%|▍                        | 17/1000 [00:52<49:30,  3.02s/it]
Epoch Progress:   2%|▍                        | 18/1000 [00:55<49:25,  3.02s/it]
Epoch Progress:   2%|▍                        | 19/1000 [00:58<49:21,  3.02s/it]
Epoch Progress:   2%|▌                        | 20/1000 [01:01<49:22,  3.02s/it]Dataset Size: 10%, Epoch: 21, Train Loss: 10.860271606445313, Val Loss: 10.872394202591536

Epoch Progress:   2%|▌                        | 21/1000 [01:04<49:17,  3.02s/it]
Epoch Progress:   2%|▌                        | 22/1000 [01:07<49:17,  3.02s/it]
Epoch Progress:   2%|▌                        | 23/1000 [01:10<49:12,  3.02s/it]
Epoch Progress:   2%|▌                        | 24/1000 [01:13<49:07,  3.02s/it]
Epoch Progress:   2%|▋                        | 25/1000 [01:16<49:02,  3.02s/it]
Epoch Progress:   3%|▋                        | 26/1000 [01:19<48:58,  3.02s/it]
Epoch Progress:   3%|▋                        | 27/1000 [01:22<48:55,  3.02s/it]
Epoch Progress:   3%|▋                        | 28/1000 [01:25<48:51,  3.02s/it]
Epoch Progress:   3%|▋                        | 29/1000 [01:28<48:48,  3.02s/it]
Epoch Progress:   3%|▊                        | 30/1000 [01:31<48:48,  3.02s/it]Dataset Size: 10%, Epoch: 31, Train Loss: 10.786469968159993, Val Loss: 10.815416459913378

Epoch Progress:   3%|▊                        | 31/1000 [01:34<48:45,  3.02s/it]
Epoch Progress:   3%|▊                        | 32/1000 [01:37<48:47,  3.02s/it]
Epoch Progress:   3%|▊                        | 33/1000 [01:40<48:41,  3.02s/it]
Epoch Progress:   3%|▊                        | 34/1000 [01:43<48:36,  3.02s/it]
Epoch Progress:   4%|▉                        | 35/1000 [01:46<48:32,  3.02s/it]
Epoch Progress:   4%|▉                        | 36/1000 [01:49<48:28,  3.02s/it]
Epoch Progress:   4%|▉                        | 37/1000 [01:52<48:27,  3.02s/it]
Epoch Progress:   4%|▉                        | 38/1000 [01:55<48:23,  3.02s/it]
Epoch Progress:   4%|▉                        | 39/1000 [01:58<48:23,  3.02s/it]
Epoch Progress:   4%|█                        | 40/1000 [02:01<48:22,  3.02s/it]Dataset Size: 10%, Epoch: 41, Train Loss: 10.701787096659343, Val Loss: 10.757361734068239

Epoch Progress:   4%|█                        | 41/1000 [02:04<48:16,  3.02s/it]
Epoch Progress:   4%|█                        | 42/1000 [02:07<48:14,  3.02s/it]
Epoch Progress:   4%|█                        | 43/1000 [02:10<48:14,  3.02s/it]
Epoch Progress:   4%|█                        | 44/1000 [02:13<48:10,  3.02s/it]
Epoch Progress:   4%|█▏                       | 45/1000 [02:16<48:04,  3.02s/it]
Epoch Progress:   5%|█▏                       | 46/1000 [02:19<47:59,  3.02s/it]
Epoch Progress:   5%|█▏                       | 47/1000 [02:22<47:57,  3.02s/it]
Epoch Progress:   5%|█▏                       | 48/1000 [02:25<47:54,  3.02s/it]
Epoch Progress:   5%|█▏                       | 49/1000 [02:28<47:50,  3.02s/it]
Epoch Progress:   5%|█▎                       | 50/1000 [02:31<47:48,  3.02s/it]Dataset Size: 10%, Epoch: 51, Train Loss: 10.510086237589519, Val Loss: 10.62926027372286

Epoch Progress:   5%|█▎                       | 51/1000 [02:34<47:42,  3.02s/it]
Epoch Progress:   5%|█▎                       | 52/1000 [02:37<47:38,  3.02s/it]
Epoch Progress:   5%|█▎                       | 53/1000 [02:40<47:43,  3.02s/it]
Epoch Progress:   5%|█▎                       | 54/1000 [02:43<47:37,  3.02s/it]
Epoch Progress:   6%|█▍                       | 55/1000 [02:46<47:31,  3.02s/it]
Epoch Progress:   6%|█▍                       | 56/1000 [02:49<47:26,  3.02s/it]
Epoch Progress:   6%|█▍                       | 57/1000 [02:52<47:25,  3.02s/it]
Epoch Progress:   6%|█▍                       | 58/1000 [02:55<47:22,  3.02s/it]
Epoch Progress:   6%|█▍                       | 59/1000 [02:59<47:17,  3.02s/it]
Epoch Progress:   6%|█▌                       | 60/1000 [03:02<47:15,  3.02s/it]Dataset Size: 10%, Epoch: 61, Train Loss: 10.218915774027506, Val Loss: 10.452091353280204

Epoch Progress:   6%|█▌                       | 61/1000 [03:05<47:13,  3.02s/it]
Epoch Progress:   6%|█▌                       | 62/1000 [03:08<47:13,  3.02s/it]
Epoch Progress:   6%|█▌                       | 63/1000 [03:11<47:11,  3.02s/it]
Epoch Progress:   6%|█▌                       | 64/1000 [03:14<47:11,  3.03s/it]
Epoch Progress:   6%|█▋                       | 65/1000 [03:17<47:05,  3.02s/it]
Epoch Progress:   7%|█▋                       | 66/1000 [03:20<46:59,  3.02s/it]
Epoch Progress:   7%|█▋                       | 67/1000 [03:23<46:57,  3.02s/it]
Epoch Progress:   7%|█▋                       | 68/1000 [03:26<46:52,  3.02s/it]
Epoch Progress:   7%|█▋                       | 69/1000 [03:29<46:48,  3.02s/it]
Epoch Progress:   7%|█▊                       | 70/1000 [03:32<46:47,  3.02s/it]Dataset Size: 10%, Epoch: 71, Train Loss: 10.02142152150472, Val Loss: 10.35897699579016

Epoch Progress:   7%|█▊                       | 71/1000 [03:35<46:41,  3.02s/it]
Epoch Progress:   7%|█▊                       | 72/1000 [03:38<46:35,  3.01s/it]
Epoch Progress:   7%|█▊                       | 73/1000 [03:41<46:34,  3.01s/it]
Epoch Progress:   7%|█▊                       | 74/1000 [03:44<46:36,  3.02s/it]
Epoch Progress:   8%|█▉                       | 75/1000 [03:47<46:33,  3.02s/it]
Epoch Progress:   8%|█▉                       | 76/1000 [03:50<46:27,  3.02s/it]
Epoch Progress:   8%|█▉                       | 77/1000 [03:53<46:25,  3.02s/it]
Epoch Progress:   8%|█▉                       | 78/1000 [03:56<46:18,  3.01s/it]
Epoch Progress:   8%|█▉                       | 79/1000 [03:59<46:14,  3.01s/it]
Epoch Progress:   8%|██                       | 80/1000 [04:02<46:11,  3.01s/it]Dataset Size: 10%, Epoch: 81, Train Loss: 9.935199559529622, Val Loss: 10.339310856608602

Epoch Progress:   8%|██                       | 81/1000 [04:05<46:09,  3.01s/it]
Epoch Progress:   8%|██                       | 82/1000 [04:08<46:09,  3.02s/it]
Epoch Progress:   8%|██                       | 83/1000 [04:11<46:11,  3.02s/it]
Epoch Progress:   8%|██                       | 84/1000 [04:14<46:06,  3.02s/it]
Epoch Progress:   8%|██▏                      | 85/1000 [04:17<46:03,  3.02s/it]
Epoch Progress:   9%|██▏                      | 86/1000 [04:20<45:56,  3.02s/it]
Epoch Progress:   9%|██▏                      | 87/1000 [04:23<45:55,  3.02s/it]
Epoch Progress:   9%|██▏                      | 88/1000 [04:26<45:50,  3.02s/it]
Epoch Progress:   9%|██▏                      | 89/1000 [04:29<45:46,  3.01s/it]
Epoch Progress:   9%|██▎                      | 90/1000 [04:32<45:46,  3.02s/it]Dataset Size: 10%, Epoch: 91, Train Loss: 9.87162556966146, Val Loss: 10.31418666591892

Epoch Progress:   9%|██▎                      | 91/1000 [04:35<45:40,  3.01s/it]
Epoch Progress:   9%|██▎                      | 92/1000 [04:38<45:36,  3.01s/it]
Epoch Progress:   9%|██▎                      | 93/1000 [04:41<45:38,  3.02s/it]
Epoch Progress:   9%|██▎                      | 94/1000 [04:44<45:32,  3.02s/it]
Epoch Progress:  10%|██▍                      | 95/1000 [04:47<45:36,  3.02s/it]
Epoch Progress:  10%|██▍                      | 96/1000 [04:50<45:31,  3.02s/it]
Epoch Progress:  10%|██▍                      | 97/1000 [04:53<45:28,  3.02s/it]
Epoch Progress:  10%|██▍                      | 98/1000 [04:56<45:21,  3.02s/it]
Epoch Progress:  10%|██▍                      | 99/1000 [04:59<45:14,  3.01s/it]
Epoch Progress:  10%|██▍                     | 100/1000 [05:02<45:14,  3.02s/it]Dataset Size: 10%, Epoch: 101, Train Loss: 9.808377532958984, Val Loss: 10.296540198388037

Epoch Progress:  10%|██▍                     | 101/1000 [05:05<45:10,  3.02s/it]
Epoch Progress:  10%|██▍                     | 102/1000 [05:08<45:08,  3.02s/it]
Epoch Progress:  10%|██▍                     | 103/1000 [05:11<45:08,  3.02s/it]
Epoch Progress:  10%|██▍                     | 104/1000 [05:14<45:04,  3.02s/it]
Epoch Progress:  10%|██▌                     | 105/1000 [05:17<45:02,  3.02s/it]
Epoch Progress:  11%|██▌                     | 106/1000 [05:20<45:01,  3.02s/it]
Epoch Progress:  11%|██▌                     | 107/1000 [05:23<44:58,  3.02s/it]
Epoch Progress:  11%|██▌                     | 108/1000 [05:26<44:54,  3.02s/it]
Epoch Progress:  11%|██▌                     | 109/1000 [05:29<44:49,  3.02s/it]
Epoch Progress:  11%|██▋                     | 110/1000 [05:32<44:46,  3.02s/it]Dataset Size: 10%, Epoch: 111, Train Loss: 9.768453381856283, Val Loss: 10.27700155431574

Epoch Progress:  11%|██▋                     | 111/1000 [05:35<44:41,  3.02s/it]
Epoch Progress:  11%|██▋                     | 112/1000 [05:38<44:38,  3.02s/it]
Epoch Progress:  11%|██▋                     | 113/1000 [05:41<44:36,  3.02s/it]
Epoch Progress:  11%|██▋                     | 114/1000 [05:44<44:32,  3.02s/it]
Epoch Progress:  12%|██▊                     | 115/1000 [05:47<44:29,  3.02s/it]
Epoch Progress:  12%|██▊                     | 116/1000 [05:51<44:37,  3.03s/it]
Epoch Progress:  12%|██▊                     | 117/1000 [05:54<44:35,  3.03s/it]
Epoch Progress:  12%|██▊                     | 118/1000 [05:57<44:29,  3.03s/it]
Epoch Progress:  12%|██▊                     | 119/1000 [06:00<44:24,  3.02s/it]
Epoch Progress:  12%|██▉                     | 120/1000 [06:03<44:20,  3.02s/it]Dataset Size: 10%, Epoch: 121, Train Loss: 9.727108459472657, Val Loss: 10.262277417368702

Epoch Progress:  12%|██▉                     | 121/1000 [06:06<44:15,  3.02s/it]
Epoch Progress:  12%|██▉                     | 122/1000 [06:09<44:11,  3.02s/it]
Epoch Progress:  12%|██▉                     | 123/1000 [06:12<44:09,  3.02s/it]
Epoch Progress:  12%|██▉                     | 124/1000 [06:15<44:03,  3.02s/it]
Epoch Progress:  12%|███                     | 125/1000 [06:18<43:59,  3.02s/it]
Epoch Progress:  13%|███                     | 126/1000 [06:21<44:01,  3.02s/it]
Epoch Progress:  13%|███                     | 127/1000 [06:24<44:02,  3.03s/it]
Epoch Progress:  13%|███                     | 128/1000 [06:27<43:56,  3.02s/it]
Epoch Progress:  13%|███                     | 129/1000 [06:30<43:51,  3.02s/it]
Epoch Progress:  13%|███                     | 130/1000 [06:33<43:46,  3.02s/it]Dataset Size: 10%, Epoch: 131, Train Loss: 9.696671613057454, Val Loss: 10.249474277744046

Epoch Progress:  13%|███▏                    | 131/1000 [06:36<43:43,  3.02s/it]
Epoch Progress:  13%|███▏                    | 132/1000 [06:39<43:38,  3.02s/it]
Epoch Progress:  13%|███▏                    | 133/1000 [06:42<43:35,  3.02s/it]
Epoch Progress:  13%|███▏                    | 134/1000 [06:45<43:31,  3.02s/it]
Epoch Progress:  14%|███▏                    | 135/1000 [06:48<43:25,  3.01s/it]
Epoch Progress:  14%|███▎                    | 136/1000 [06:51<43:24,  3.01s/it]
Epoch Progress:  14%|███▎                    | 137/1000 [06:54<43:27,  3.02s/it]
Epoch Progress:  14%|███▎                    | 138/1000 [06:57<43:23,  3.02s/it]
Epoch Progress:  14%|███▎                    | 139/1000 [07:00<43:20,  3.02s/it]
Epoch Progress:  14%|███▎                    | 140/1000 [07:03<43:17,  3.02s/it]Dataset Size: 10%, Epoch: 141, Train Loss: 9.676832173665364, Val Loss: 10.247894076557902

Epoch Progress:  14%|███▍                    | 141/1000 [07:06<43:11,  3.02s/it]
Epoch Progress:  14%|███▍                    | 142/1000 [07:09<43:08,  3.02s/it]
Epoch Progress:  14%|███▍                    | 143/1000 [07:12<43:05,  3.02s/it]
Epoch Progress:  14%|███▍                    | 144/1000 [07:15<43:00,  3.01s/it]
Epoch Progress:  14%|███▍                    | 145/1000 [07:18<42:57,  3.02s/it]
Epoch Progress:  15%|███▌                    | 146/1000 [07:21<42:58,  3.02s/it]
Epoch Progress:  15%|███▌                    | 147/1000 [07:24<42:53,  3.02s/it]
Epoch Progress:  15%|███▌                    | 148/1000 [07:27<42:51,  3.02s/it]
Epoch Progress:  15%|███▌                    | 149/1000 [07:30<42:47,  3.02s/it]
Epoch Progress:  15%|███▌                    | 150/1000 [07:33<42:44,  3.02s/it]Dataset Size: 10%, Epoch: 151, Train Loss: 9.640760269165039, Val Loss: 10.232338397533862

Epoch Progress:  15%|███▌                    | 151/1000 [07:36<42:39,  3.01s/it]
Epoch Progress:  15%|███▋                    | 152/1000 [07:39<42:33,  3.01s/it]
Epoch Progress:  15%|███▋                    | 153/1000 [07:42<42:32,  3.01s/it]
Epoch Progress:  15%|███▋                    | 154/1000 [07:45<42:29,  3.01s/it]
Epoch Progress:  16%|███▋                    | 155/1000 [07:48<42:26,  3.01s/it]
Epoch Progress:  16%|███▋                    | 156/1000 [07:51<42:26,  3.02s/it]
Epoch Progress:  16%|███▊                    | 157/1000 [07:54<42:21,  3.02s/it]
Epoch Progress:  16%|███▊                    | 158/1000 [07:57<42:25,  3.02s/it]
Epoch Progress:  16%|███▊                    | 159/1000 [08:00<42:20,  3.02s/it]
Epoch Progress:  16%|███▊                    | 160/1000 [08:03<42:14,  3.02s/it]Dataset Size: 10%, Epoch: 161, Train Loss: 9.623952484130859, Val Loss: 10.224081051814093

Epoch Progress:  16%|███▊                    | 161/1000 [08:06<42:10,  3.02s/it]
Epoch Progress:  16%|███▉                    | 162/1000 [08:09<42:09,  3.02s/it]
Epoch Progress:  16%|███▉                    | 163/1000 [08:12<42:06,  3.02s/it]
Epoch Progress:  16%|███▉                    | 164/1000 [08:15<42:01,  3.02s/it]
Epoch Progress:  16%|███▉                    | 165/1000 [08:18<41:54,  3.01s/it]
Epoch Progress:  17%|███▉                    | 166/1000 [08:21<41:53,  3.01s/it]
Epoch Progress:  17%|████                    | 167/1000 [08:24<41:49,  3.01s/it]
Epoch Progress:  17%|████                    | 168/1000 [08:27<41:46,  3.01s/it]
Epoch Progress:  17%|████                    | 169/1000 [08:30<41:47,  3.02s/it]
Epoch Progress:  17%|████                    | 170/1000 [08:34<41:44,  3.02s/it]Dataset Size: 10%, Epoch: 171, Train Loss: 9.61196445465088, Val Loss: 10.225855282374791

Epoch Progress:  17%|████                    | 171/1000 [08:37<41:40,  3.02s/it]
Epoch Progress:  17%|████▏                   | 172/1000 [08:40<41:36,  3.01s/it]
Epoch Progress:  17%|████▏                   | 173/1000 [08:43<41:33,  3.02s/it]
Epoch Progress:  17%|████▏                   | 174/1000 [08:46<41:31,  3.02s/it]
Epoch Progress:  18%|████▏                   | 175/1000 [08:49<41:27,  3.01s/it]
Epoch Progress:  18%|████▏                   | 176/1000 [08:52<41:24,  3.02s/it]
Epoch Progress:  18%|████▏                   | 177/1000 [08:55<41:19,  3.01s/it]
Epoch Progress:  18%|████▎                   | 178/1000 [08:58<41:15,  3.01s/it]
Epoch Progress:  18%|████▎                   | 179/1000 [09:01<41:20,  3.02s/it]
Epoch Progress:  18%|████▎                   | 180/1000 [09:04<41:17,  3.02s/it]Dataset Size: 10%, Epoch: 181, Train Loss: 9.596492182413737, Val Loss: 10.220311363022049

Epoch Progress:  18%|████▎                   | 181/1000 [09:07<41:12,  3.02s/it]
Epoch Progress:  18%|████▎                   | 182/1000 [09:10<41:08,  3.02s/it]
Epoch Progress:  18%|████▍                   | 183/1000 [09:13<41:03,  3.02s/it]
Epoch Progress:  18%|████▍                   | 184/1000 [09:16<40:57,  3.01s/it]
Epoch Progress:  18%|████▍                   | 185/1000 [09:19<40:53,  3.01s/it]
Epoch Progress:  19%|████▍                   | 186/1000 [09:22<40:53,  3.01s/it]
Epoch Progress:  19%|████▍                   | 187/1000 [09:25<40:48,  3.01s/it]
Epoch Progress:  19%|████▌                   | 188/1000 [09:28<40:43,  3.01s/it]
Epoch Progress:  19%|████▌                   | 189/1000 [09:31<40:42,  3.01s/it]
Epoch Progress:  19%|████▌                   | 190/1000 [09:34<40:40,  3.01s/it]Dataset Size: 10%, Epoch: 191, Train Loss: 9.584501457214355, Val Loss: 10.21820136478969

Epoch Progress:  19%|████▌                   | 191/1000 [09:37<40:37,  3.01s/it]
Epoch Progress:  19%|████▌                   | 192/1000 [09:40<40:38,  3.02s/it]
Epoch Progress:  19%|████▋                   | 193/1000 [09:43<40:35,  3.02s/it]
Epoch Progress:  19%|████▋                   | 194/1000 [09:46<40:29,  3.01s/it]
Epoch Progress:  20%|████▋                   | 195/1000 [09:49<40:27,  3.02s/it]
Epoch Progress:  20%|████▋                   | 196/1000 [09:52<40:24,  3.02s/it]
Epoch Progress:  20%|████▋                   | 197/1000 [09:55<40:19,  3.01s/it]
Epoch Progress:  20%|████▊                   | 198/1000 [09:58<40:16,  3.01s/it]
Epoch Progress:  20%|████▊                   | 199/1000 [10:01<40:15,  3.02s/it]
Epoch Progress:  20%|████▊                   | 200/1000 [10:04<40:14,  3.02s/it]Dataset Size: 10%, Epoch: 201, Train Loss: 9.569962641398112, Val Loss: 10.217459046995485

Epoch Progress:  20%|████▊                   | 201/1000 [10:07<40:12,  3.02s/it]
Epoch Progress:  20%|████▊                   | 202/1000 [10:10<40:08,  3.02s/it]
Epoch Progress:  20%|████▊                   | 203/1000 [10:13<40:06,  3.02s/it]
Epoch Progress:  20%|████▉                   | 204/1000 [10:16<40:01,  3.02s/it]
Epoch Progress:  20%|████▉                   | 205/1000 [10:19<39:57,  3.02s/it]
Epoch Progress:  21%|████▉                   | 206/1000 [10:22<39:54,  3.02s/it]
Epoch Progress:  21%|████▉                   | 207/1000 [10:25<39:50,  3.01s/it]
Epoch Progress:  21%|████▉                   | 208/1000 [10:28<39:47,  3.01s/it]
Epoch Progress:  21%|█████                   | 209/1000 [10:31<39:46,  3.02s/it]
Epoch Progress:  21%|█████                   | 210/1000 [10:34<39:42,  3.02s/it]Dataset Size: 10%, Epoch: 211, Train Loss: 9.567035306294759, Val Loss: 10.2088599266944

Epoch Progress:  21%|█████                   | 211/1000 [10:37<39:41,  3.02s/it]
Epoch Progress:  21%|█████                   | 212/1000 [10:40<39:37,  3.02s/it]
Epoch Progress:  21%|█████                   | 213/1000 [10:43<39:33,  3.02s/it]
Epoch Progress:  21%|█████▏                  | 214/1000 [10:46<39:27,  3.01s/it]
Epoch Progress:  22%|█████▏                  | 215/1000 [10:49<39:23,  3.01s/it]
Epoch Progress:  22%|█████▏                  | 216/1000 [10:52<39:22,  3.01s/it]
Epoch Progress:  22%|█████▏                  | 217/1000 [10:55<39:19,  3.01s/it]
Epoch Progress:  22%|█████▏                  | 218/1000 [10:58<39:16,  3.01s/it]
Epoch Progress:  22%|█████▎                  | 219/1000 [11:01<39:15,  3.02s/it]
Epoch Progress:  22%|█████▎                  | 220/1000 [11:04<39:13,  3.02s/it]Dataset Size: 10%, Epoch: 221, Train Loss: 9.551784706115722, Val Loss: 10.206955216147684

Epoch Progress:  22%|█████▎                  | 221/1000 [11:07<39:12,  3.02s/it]
Epoch Progress:  22%|█████▎                  | 222/1000 [11:10<39:10,  3.02s/it]
Epoch Progress:  22%|█████▎                  | 223/1000 [11:13<39:05,  3.02s/it]
Epoch Progress:  22%|█████▍                  | 224/1000 [11:16<38:59,  3.02s/it]
Epoch Progress:  22%|█████▍                  | 225/1000 [11:19<38:56,  3.01s/it]
Epoch Progress:  23%|█████▍                  | 226/1000 [11:22<38:55,  3.02s/it]
Epoch Progress:  23%|█████▍                  | 227/1000 [11:25<38:50,  3.01s/it]
Epoch Progress:  23%|█████▍                  | 228/1000 [11:28<38:48,  3.02s/it]
Epoch Progress:  23%|█████▍                  | 229/1000 [11:31<38:45,  3.02s/it]
Epoch Progress:  23%|█████▌                  | 230/1000 [11:34<38:40,  3.01s/it]Dataset Size: 10%, Epoch: 231, Train Loss: 9.543075485229492, Val Loss: 10.200785488277287

Epoch Progress:  23%|█████▌                  | 231/1000 [11:37<38:36,  3.01s/it]
Epoch Progress:  23%|█████▌                  | 232/1000 [11:40<38:39,  3.02s/it]
Epoch Progress:  23%|█████▌                  | 233/1000 [11:43<38:33,  3.02s/it]
Epoch Progress:  23%|█████▌                  | 234/1000 [11:46<38:29,  3.02s/it]
Epoch Progress:  24%|█████▋                  | 235/1000 [11:49<38:23,  3.01s/it]
Epoch Progress:  24%|█████▋                  | 236/1000 [11:53<38:22,  3.01s/it]
Epoch Progress:  24%|█████▋                  | 237/1000 [11:56<38:19,  3.01s/it]
Epoch Progress:  24%|█████▋                  | 238/1000 [11:59<38:15,  3.01s/it]
Epoch Progress:  24%|█████▋                  | 239/1000 [12:02<38:14,  3.01s/it]
Epoch Progress:  24%|█████▊                  | 240/1000 [12:05<38:09,  3.01s/it]Dataset Size: 10%, Epoch: 241, Train Loss: 9.534119962056478, Val Loss: 10.20068022492644

Epoch Progress:  24%|█████▊                  | 241/1000 [12:08<38:05,  3.01s/it]
Epoch Progress:  24%|█████▊                  | 242/1000 [12:11<38:07,  3.02s/it]
Epoch Progress:  24%|█████▊                  | 243/1000 [12:14<38:07,  3.02s/it]
Epoch Progress:  24%|█████▊                  | 244/1000 [12:17<38:03,  3.02s/it]
Epoch Progress:  24%|█████▉                  | 245/1000 [12:20<37:57,  3.02s/it]
Epoch Progress:  25%|█████▉                  | 246/1000 [12:23<37:58,  3.02s/it]
Epoch Progress:  25%|█████▉                  | 247/1000 [12:26<37:54,  3.02s/it]
Epoch Progress:  25%|█████▉                  | 248/1000 [12:29<37:52,  3.02s/it]
Epoch Progress:  25%|█████▉                  | 249/1000 [12:32<37:47,  3.02s/it]
Epoch Progress:  25%|██████                  | 250/1000 [12:35<37:43,  3.02s/it]Dataset Size: 10%, Epoch: 251, Train Loss: 9.51860590616862, Val Loss: 10.1948282687695

Epoch Progress:  25%|██████                  | 251/1000 [12:38<37:38,  3.02s/it]
Epoch Progress:  25%|██████                  | 252/1000 [12:41<37:36,  3.02s/it]
Epoch Progress:  25%|██████                  | 253/1000 [12:44<37:34,  3.02s/it]
Epoch Progress:  25%|██████                  | 254/1000 [12:47<37:28,  3.01s/it]
Epoch Progress:  26%|██████                  | 255/1000 [12:50<37:23,  3.01s/it]
Epoch Progress:  26%|██████▏                 | 256/1000 [12:53<37:21,  3.01s/it]
Epoch Progress:  26%|██████▏                 | 257/1000 [12:56<37:18,  3.01s/it]
Epoch Progress:  26%|██████▏                 | 258/1000 [12:59<37:15,  3.01s/it]
Epoch Progress:  26%|██████▏                 | 259/1000 [13:02<37:14,  3.02s/it]
Epoch Progress:  26%|██████▏                 | 260/1000 [13:05<37:11,  3.02s/it]Dataset Size: 10%, Epoch: 261, Train Loss: 9.513001149495443, Val Loss: 10.19400540884439

Epoch Progress:  26%|██████▎                 | 261/1000 [13:08<37:06,  3.01s/it]
Epoch Progress:  26%|██████▎                 | 262/1000 [13:11<37:09,  3.02s/it]
Epoch Progress:  26%|██████▎                 | 263/1000 [13:14<37:07,  3.02s/it]
Epoch Progress:  26%|██████▎                 | 264/1000 [13:17<37:03,  3.02s/it]
Epoch Progress:  26%|██████▎                 | 265/1000 [13:20<36:58,  3.02s/it]
Epoch Progress:  27%|██████▍                 | 266/1000 [13:23<36:55,  3.02s/it]
Epoch Progress:  27%|██████▍                 | 267/1000 [13:26<36:50,  3.02s/it]
Epoch Progress:  27%|██████▍                 | 268/1000 [13:29<36:45,  3.01s/it]
Epoch Progress:  27%|██████▍                 | 269/1000 [13:32<36:42,  3.01s/it]
Epoch Progress:  27%|██████▍                 | 270/1000 [13:35<36:38,  3.01s/it]Dataset Size: 10%, Epoch: 271, Train Loss: 9.506695454915365, Val Loss: 10.188781676354346

Epoch Progress:  27%|██████▌                 | 271/1000 [13:38<36:35,  3.01s/it]
Epoch Progress:  27%|██████▌                 | 272/1000 [13:41<36:36,  3.02s/it]
Epoch Progress:  27%|██████▌                 | 273/1000 [13:44<36:31,  3.01s/it]
Epoch Progress:  27%|██████▌                 | 274/1000 [13:47<36:32,  3.02s/it]
Epoch Progress:  28%|██████▌                 | 275/1000 [13:50<36:27,  3.02s/it]
Epoch Progress:  28%|██████▌                 | 276/1000 [13:53<36:24,  3.02s/it]
Epoch Progress:  28%|██████▋                 | 277/1000 [13:56<36:19,  3.01s/it]
Epoch Progress:  28%|██████▋                 | 278/1000 [13:59<36:14,  3.01s/it]
Epoch Progress:  28%|██████▋                 | 279/1000 [14:02<36:12,  3.01s/it]
Epoch Progress:  28%|██████▋                 | 280/1000 [14:05<36:11,  3.02s/it]Dataset Size: 10%, Epoch: 281, Train Loss: 9.494052340189615, Val Loss: 10.192359911931025

Epoch Progress:  28%|██████▋                 | 281/1000 [14:08<36:08,  3.02s/it]
Epoch Progress:  28%|██████▊                 | 282/1000 [14:11<36:09,  3.02s/it]
Epoch Progress:  28%|██████▊                 | 283/1000 [14:14<36:04,  3.02s/it]
Epoch Progress:  28%|██████▊                 | 284/1000 [14:17<36:00,  3.02s/it]
Epoch Progress:  28%|██████▊                 | 285/1000 [14:20<35:59,  3.02s/it]
Epoch Progress:  29%|██████▊                 | 286/1000 [14:23<35:57,  3.02s/it]
Epoch Progress:  29%|██████▉                 | 287/1000 [14:26<35:52,  3.02s/it]
Epoch Progress:  29%|██████▉                 | 288/1000 [14:29<35:47,  3.02s/it]
Epoch Progress:  29%|██████▉                 | 289/1000 [14:32<35:45,  3.02s/it]
Epoch Progress:  29%|██████▉                 | 290/1000 [14:35<35:40,  3.02s/it]Dataset Size: 10%, Epoch: 291, Train Loss: 9.487770334879558, Val Loss: 10.191109236184653

Epoch Progress:  29%|██████▉                 | 291/1000 [14:38<35:36,  3.01s/it]
Epoch Progress:  29%|███████                 | 292/1000 [14:41<35:36,  3.02s/it]
Epoch Progress:  29%|███████                 | 293/1000 [14:44<35:30,  3.01s/it]
Epoch Progress:  29%|███████                 | 294/1000 [14:47<35:26,  3.01s/it]
Epoch Progress:  30%|███████                 | 295/1000 [14:50<35:26,  3.02s/it]
Epoch Progress:  30%|███████                 | 296/1000 [14:54<35:24,  3.02s/it]
Epoch Progress:  30%|███████▏                | 297/1000 [14:57<35:19,  3.02s/it]
Epoch Progress:  30%|███████▏                | 298/1000 [15:00<35:15,  3.01s/it]
Epoch Progress:  30%|███████▏                | 299/1000 [15:03<35:12,  3.01s/it]
Epoch Progress:  30%|███████▏                | 300/1000 [15:06<35:10,  3.01s/it]Dataset Size: 10%, Epoch: 301, Train Loss: 9.481020965576171, Val Loss: 10.190443843990177

Epoch Progress:  30%|███████▏                | 301/1000 [15:09<35:07,  3.01s/it]
Epoch Progress:  30%|███████▏                | 302/1000 [15:12<35:07,  3.02s/it]
Epoch Progress:  30%|███████▎                | 303/1000 [15:15<35:03,  3.02s/it]
Epoch Progress:  30%|███████▎                | 304/1000 [15:18<34:57,  3.01s/it]
Epoch Progress:  30%|███████▎                | 305/1000 [15:21<34:57,  3.02s/it]
Epoch Progress:  31%|███████▎                | 306/1000 [15:24<34:58,  3.02s/it]
Epoch Progress:  31%|███████▎                | 307/1000 [15:27<34:53,  3.02s/it]
Epoch Progress:  31%|███████▍                | 308/1000 [15:30<34:47,  3.02s/it]
Epoch Progress:  31%|███████▍                | 309/1000 [15:33<34:47,  3.02s/it]
Epoch Progress:  31%|███████▍                | 310/1000 [15:36<34:43,  3.02s/it]Dataset Size: 10%, Epoch: 311, Train Loss: 9.476043752034505, Val Loss: 10.186462018396947

Epoch Progress:  31%|███████▍                | 311/1000 [15:39<34:38,  3.02s/it]
Epoch Progress:  31%|███████▍                | 312/1000 [15:42<34:37,  3.02s/it]
Epoch Progress:  31%|███████▌                | 313/1000 [15:45<34:35,  3.02s/it]
Epoch Progress:  31%|███████▌                | 314/1000 [15:48<34:30,  3.02s/it]
Epoch Progress:  32%|███████▌                | 315/1000 [15:51<34:28,  3.02s/it]
Epoch Progress:  32%|███████▌                | 316/1000 [15:54<34:30,  3.03s/it]
Epoch Progress:  32%|███████▌                | 317/1000 [15:57<34:25,  3.02s/it]
Epoch Progress:  32%|███████▋                | 318/1000 [16:00<34:19,  3.02s/it]
Epoch Progress:  32%|███████▋                | 319/1000 [16:03<34:15,  3.02s/it]
Epoch Progress:  32%|███████▋                | 320/1000 [16:06<34:11,  3.02s/it]Dataset Size: 10%, Epoch: 321, Train Loss: 9.474281717936197, Val Loss: 10.186976581424862

Epoch Progress:  32%|███████▋                | 321/1000 [16:09<34:08,  3.02s/it]
Epoch Progress:  32%|███████▋                | 322/1000 [16:12<34:04,  3.02s/it]
Epoch Progress:  32%|███████▊                | 323/1000 [16:15<34:03,  3.02s/it]
Epoch Progress:  32%|███████▊                | 324/1000 [16:18<34:00,  3.02s/it]
Epoch Progress:  32%|███████▊                | 325/1000 [16:21<33:59,  3.02s/it]
Epoch Progress:  33%|███████▊                | 326/1000 [16:24<33:55,  3.02s/it]
Epoch Progress:  33%|███████▊                | 327/1000 [16:27<33:51,  3.02s/it]
Epoch Progress:  33%|███████▊                | 328/1000 [16:30<33:46,  3.02s/it]
Epoch Progress:  33%|███████▉                | 329/1000 [16:33<33:43,  3.02s/it]
Epoch Progress:  33%|███████▉                | 330/1000 [16:36<33:39,  3.01s/it]Dataset Size: 10%, Epoch: 331, Train Loss: 9.464540239969889, Val Loss: 10.181162338752252

Epoch Progress:  33%|███████▉                | 331/1000 [16:39<33:36,  3.01s/it]
Epoch Progress:  33%|███████▉                | 332/1000 [16:42<33:34,  3.02s/it]
Epoch Progress:  33%|███████▉                | 333/1000 [16:45<33:30,  3.01s/it]
Epoch Progress:  33%|████████                | 334/1000 [16:48<33:27,  3.01s/it]
Epoch Progress:  34%|████████                | 335/1000 [16:51<33:26,  3.02s/it]
Epoch Progress:  34%|████████                | 336/1000 [16:54<33:21,  3.01s/it]
Epoch Progress:  34%|████████                | 337/1000 [16:57<33:21,  3.02s/it]
Epoch Progress:  34%|████████                | 338/1000 [17:00<33:17,  3.02s/it]
Epoch Progress:  34%|████████▏               | 339/1000 [17:03<33:15,  3.02s/it]
Epoch Progress:  34%|████████▏               | 340/1000 [17:06<33:10,  3.02s/it]Dataset Size: 10%, Epoch: 341, Train Loss: 9.459121348063151, Val Loss: 10.183110608683004

Epoch Progress:  34%|████████▏               | 341/1000 [17:09<33:08,  3.02s/it]
Epoch Progress:  34%|████████▏               | 342/1000 [17:12<33:05,  3.02s/it]
Epoch Progress:  34%|████████▏               | 343/1000 [17:15<33:01,  3.02s/it]
Epoch Progress:  34%|████████▎               | 344/1000 [17:18<32:58,  3.02s/it]
Epoch Progress:  34%|████████▎               | 345/1000 [17:21<32:57,  3.02s/it]
Epoch Progress:  35%|████████▎               | 346/1000 [17:24<32:52,  3.02s/it]
Epoch Progress:  35%|████████▎               | 347/1000 [17:27<32:47,  3.01s/it]
Epoch Progress:  35%|████████▎               | 348/1000 [17:30<32:46,  3.02s/it]
Epoch Progress:  35%|████████▍               | 349/1000 [17:33<32:43,  3.02s/it]
Epoch Progress:  35%|████████▍               | 350/1000 [17:36<32:38,  3.01s/it]Dataset Size: 10%, Epoch: 351, Train Loss: 9.454736035664876, Val Loss: 10.18539997199913

Epoch Progress:  35%|████████▍               | 351/1000 [17:39<32:35,  3.01s/it]
Epoch Progress:  35%|████████▍               | 352/1000 [17:42<32:32,  3.01s/it]
Epoch Progress:  35%|████████▍               | 353/1000 [17:45<32:30,  3.01s/it]
Epoch Progress:  35%|████████▍               | 354/1000 [17:48<32:27,  3.01s/it]
Epoch Progress:  36%|████████▌               | 355/1000 [17:52<32:24,  3.02s/it]
Epoch Progress:  36%|████████▌               | 356/1000 [17:55<32:21,  3.01s/it]
Epoch Progress:  36%|████████▌               | 357/1000 [17:58<32:18,  3.01s/it]
Epoch Progress:  36%|████████▌               | 358/1000 [18:01<32:20,  3.02s/it]
Epoch Progress:  36%|████████▌               | 359/1000 [18:04<32:15,  3.02s/it]
Epoch Progress:  36%|████████▋               | 360/1000 [18:07<32:10,  3.02s/it]Dataset Size: 10%, Epoch: 361, Train Loss: 9.45269442240397, Val Loss: 10.18072728986864

Epoch Progress:  36%|████████▋               | 361/1000 [18:10<32:09,  3.02s/it]
Epoch Progress:  36%|████████▋               | 362/1000 [18:13<32:05,  3.02s/it]
Epoch Progress:  36%|████████▋               | 363/1000 [18:16<32:01,  3.02s/it]
Epoch Progress:  36%|████████▋               | 364/1000 [18:19<31:56,  3.01s/it]
Epoch Progress:  36%|████████▊               | 365/1000 [18:22<31:54,  3.01s/it]
Epoch Progress:  37%|████████▊               | 366/1000 [18:25<31:49,  3.01s/it]
Epoch Progress:  37%|████████▊               | 367/1000 [18:28<31:44,  3.01s/it]
Epoch Progress:  37%|████████▊               | 368/1000 [18:31<31:43,  3.01s/it]
Epoch Progress:  37%|████████▊               | 369/1000 [18:34<31:42,  3.02s/it]
Epoch Progress:  37%|████████▉               | 370/1000 [18:37<31:38,  3.01s/it]Dataset Size: 10%, Epoch: 371, Train Loss: 9.447982991536458, Val Loss: 10.184659747334269

Epoch Progress:  37%|████████▉               | 371/1000 [18:40<31:35,  3.01s/it]
Epoch Progress:  37%|████████▉               | 372/1000 [18:43<31:34,  3.02s/it]
Epoch Progress:  37%|████████▉               | 373/1000 [18:46<31:30,  3.02s/it]
Epoch Progress:  37%|████████▉               | 374/1000 [18:49<31:27,  3.01s/it]
Epoch Progress:  38%|█████████               | 375/1000 [18:52<31:24,  3.01s/it]
Epoch Progress:  38%|█████████               | 376/1000 [18:55<31:20,  3.01s/it]
Epoch Progress:  38%|█████████               | 377/1000 [18:58<31:16,  3.01s/it]
Epoch Progress:  38%|█████████               | 378/1000 [19:01<31:14,  3.01s/it]
Epoch Progress:  38%|█████████               | 379/1000 [19:04<31:15,  3.02s/it]
Epoch Progress:  38%|█████████               | 380/1000 [19:07<31:12,  3.02s/it]Dataset Size: 10%, Epoch: 381, Train Loss: 9.438130912780762, Val Loss: 10.182001819858304

Epoch Progress:  38%|█████████▏              | 381/1000 [19:10<31:08,  3.02s/it]
Epoch Progress:  38%|█████████▏              | 382/1000 [19:13<31:07,  3.02s/it]
Epoch Progress:  38%|█████████▏              | 383/1000 [19:16<31:04,  3.02s/it]
Epoch Progress:  38%|█████████▏              | 384/1000 [19:19<31:00,  3.02s/it]
Epoch Progress:  38%|█████████▏              | 385/1000 [19:22<30:58,  3.02s/it]
Epoch Progress:  39%|█████████▎              | 386/1000 [19:25<30:53,  3.02s/it]
Epoch Progress:  39%|█████████▎              | 387/1000 [19:28<30:49,  3.02s/it]
Epoch Progress:  39%|█████████▎              | 388/1000 [19:31<30:46,  3.02s/it]
Epoch Progress:  39%|█████████▎              | 389/1000 [19:34<30:43,  3.02s/it]
Epoch Progress:  39%|█████████▎              | 390/1000 [19:37<30:42,  3.02s/it]Dataset Size: 10%, Epoch: 391, Train Loss: 9.44769078572591, Val Loss: 10.182669367109026

Epoch Progress:  39%|█████████▍              | 391/1000 [19:40<30:37,  3.02s/it]
Epoch Progress:  39%|█████████▍              | 392/1000 [19:43<30:33,  3.02s/it]
Epoch Progress:  39%|█████████▍              | 393/1000 [19:46<30:27,  3.01s/it]
Epoch Progress:  39%|█████████▍              | 394/1000 [19:49<30:24,  3.01s/it]
Epoch Progress:  40%|█████████▍              | 395/1000 [19:52<30:22,  3.01s/it]
Epoch Progress:  40%|█████████▌              | 396/1000 [19:55<30:17,  3.01s/it]
Epoch Progress:  40%|█████████▌              | 397/1000 [19:58<30:14,  3.01s/it]
Epoch Progress:  40%|█████████▌              | 398/1000 [20:01<30:13,  3.01s/it]
Epoch Progress:  40%|█████████▌              | 399/1000 [20:04<30:11,  3.01s/it]
Epoch Progress:  40%|█████████▌              | 400/1000 [20:07<30:11,  3.02s/it]Dataset Size: 10%, Epoch: 401, Train Loss: 9.438140576680501, Val Loss: 10.181039636785334

Epoch Progress:  40%|█████████▌              | 401/1000 [20:10<30:06,  3.02s/it]
Epoch Progress:  40%|█████████▋              | 402/1000 [20:13<30:04,  3.02s/it]
Epoch Progress:  40%|█████████▋              | 403/1000 [20:16<30:00,  3.02s/it]
Epoch Progress:  40%|█████████▋              | 404/1000 [20:19<29:56,  3.01s/it]
Epoch Progress:  40%|█████████▋              | 405/1000 [20:22<29:55,  3.02s/it]
Epoch Progress:  41%|█████████▋              | 406/1000 [20:25<29:51,  3.02s/it]
Epoch Progress:  41%|█████████▊              | 407/1000 [20:28<29:48,  3.02s/it]
Epoch Progress:  41%|█████████▊              | 408/1000 [20:31<29:45,  3.02s/it]
Epoch Progress:  41%|█████████▊              | 409/1000 [20:34<29:41,  3.01s/it]
Epoch Progress:  41%|█████████▊              | 410/1000 [20:37<29:37,  3.01s/it]Dataset Size: 10%, Epoch: 411, Train Loss: 9.437002983093262, Val Loss: 10.186399596078056

Epoch Progress:  41%|█████████▊              | 411/1000 [20:40<29:36,  3.02s/it]
Epoch Progress:  41%|█████████▉              | 412/1000 [20:43<29:32,  3.01s/it]
Epoch Progress:  41%|█████████▉              | 413/1000 [20:46<29:28,  3.01s/it]
Epoch Progress:  41%|█████████▉              | 414/1000 [20:49<29:25,  3.01s/it]
Epoch Progress:  42%|█████████▉              | 415/1000 [20:52<29:23,  3.01s/it]
Epoch Progress:  42%|█████████▉              | 416/1000 [20:55<29:20,  3.01s/it]
Epoch Progress:  42%|██████████              | 417/1000 [20:58<29:17,  3.01s/it]
Epoch Progress:  42%|██████████              | 418/1000 [21:01<29:15,  3.02s/it]
Epoch Progress:  42%|██████████              | 419/1000 [21:05<29:11,  3.01s/it]
Epoch Progress:  42%|██████████              | 420/1000 [21:08<29:07,  3.01s/it]Dataset Size: 10%, Epoch: 421, Train Loss: 9.43435489654541, Val Loss: 10.182176193633637

Epoch Progress:  42%|██████████              | 421/1000 [21:11<29:11,  3.03s/it]
Epoch Progress:  42%|██████████▏             | 422/1000 [21:14<29:06,  3.02s/it]
Epoch Progress:  42%|██████████▏             | 423/1000 [21:17<29:02,  3.02s/it]
Epoch Progress:  42%|██████████▏             | 424/1000 [21:20<28:59,  3.02s/it]
Epoch Progress:  42%|██████████▏             | 425/1000 [21:23<28:56,  3.02s/it]
Epoch Progress:  43%|██████████▏             | 426/1000 [21:26<28:52,  3.02s/it]
Epoch Progress:  43%|██████████▏             | 427/1000 [21:29<28:49,  3.02s/it]
Epoch Progress:  43%|██████████▎             | 428/1000 [21:32<28:47,  3.02s/it]
Epoch Progress:  43%|██████████▎             | 429/1000 [21:35<28:41,  3.01s/it]
Epoch Progress:  43%|██████████▎             | 430/1000 [21:38<28:36,  3.01s/it]Dataset Size: 10%, Epoch: 431, Train Loss: 9.423664830525716, Val Loss: 10.181682599055303

Epoch Progress:  43%|██████████▎             | 431/1000 [21:41<28:35,  3.01s/it]
Epoch Progress:  43%|██████████▎             | 432/1000 [21:44<28:33,  3.02s/it]
Epoch Progress:  43%|██████████▍             | 433/1000 [21:47<28:27,  3.01s/it]
Epoch Progress:  43%|██████████▍             | 434/1000 [21:50<28:24,  3.01s/it]
Epoch Progress:  44%|██████████▍             | 435/1000 [21:53<28:21,  3.01s/it]
Epoch Progress:  44%|██████████▍             | 436/1000 [21:56<28:17,  3.01s/it]
Epoch Progress:  44%|██████████▍             | 437/1000 [21:59<28:15,  3.01s/it]
Epoch Progress:  44%|██████████▌             | 438/1000 [22:02<28:13,  3.01s/it]
Epoch Progress:  44%|██████████▌             | 439/1000 [22:05<28:09,  3.01s/it]
Epoch Progress:  44%|██████████▌             | 440/1000 [22:08<28:05,  3.01s/it]Dataset Size: 10%, Epoch: 441, Train Loss: 9.42647169748942, Val Loss: 10.178853666627562

Epoch Progress:  44%|██████████▌             | 441/1000 [22:11<28:03,  3.01s/it]
Epoch Progress:  44%|██████████▌             | 442/1000 [22:14<28:03,  3.02s/it]
Epoch Progress:  44%|██████████▋             | 443/1000 [22:17<28:01,  3.02s/it]
Epoch Progress:  44%|██████████▋             | 444/1000 [22:20<27:57,  3.02s/it]
Epoch Progress:  44%|██████████▋             | 445/1000 [22:23<27:52,  3.01s/it]
Epoch Progress:  45%|██████████▋             | 446/1000 [22:26<27:49,  3.01s/it]
Epoch Progress:  45%|██████████▋             | 447/1000 [22:29<27:45,  3.01s/it]
Epoch Progress:  45%|██████████▊             | 448/1000 [22:32<27:43,  3.01s/it]
Epoch Progress:  45%|██████████▊             | 449/1000 [22:35<27:39,  3.01s/it]
Epoch Progress:  45%|██████████▊             | 450/1000 [22:38<27:36,  3.01s/it]Dataset Size: 10%, Epoch: 451, Train Loss: 9.428352788289388, Val Loss: 10.184675910256125

Epoch Progress:  45%|██████████▊             | 451/1000 [22:41<27:34,  3.01s/it]
Epoch Progress:  45%|██████████▊             | 452/1000 [22:44<27:31,  3.01s/it]
Epoch Progress:  45%|██████████▊             | 453/1000 [22:47<27:29,  3.02s/it]
Epoch Progress:  45%|██████████▉             | 454/1000 [22:50<27:25,  3.01s/it]
Epoch Progress:  46%|██████████▉             | 455/1000 [22:53<27:22,  3.01s/it]
Epoch Progress:  46%|██████████▉             | 456/1000 [22:56<27:18,  3.01s/it]
Epoch Progress:  46%|██████████▉             | 457/1000 [22:59<27:15,  3.01s/it]
Epoch Progress:  46%|██████████▉             | 458/1000 [23:02<27:11,  3.01s/it]
Epoch Progress:  46%|███████████             | 459/1000 [23:05<27:06,  3.01s/it]
Epoch Progress:  46%|███████████             | 460/1000 [23:08<27:03,  3.01s/it]Dataset Size: 10%, Epoch: 461, Train Loss: 9.423418388366699, Val Loss: 10.182473244605127

Epoch Progress:  46%|███████████             | 461/1000 [23:11<27:01,  3.01s/it]
Epoch Progress:  46%|███████████             | 462/1000 [23:14<26:58,  3.01s/it]
Epoch Progress:  46%|███████████             | 463/1000 [23:17<26:59,  3.02s/it]
Epoch Progress:  46%|███████████▏            | 464/1000 [23:20<26:56,  3.02s/it]
Epoch Progress:  46%|███████████▏            | 465/1000 [23:23<26:53,  3.02s/it]
Epoch Progress:  47%|███████████▏            | 466/1000 [23:26<26:49,  3.01s/it]
Epoch Progress:  47%|███████████▏            | 467/1000 [23:29<26:44,  3.01s/it]
Epoch Progress:  47%|███████████▏            | 468/1000 [23:32<26:40,  3.01s/it]
Epoch Progress:  47%|███████████▎            | 469/1000 [23:35<26:38,  3.01s/it]
Epoch Progress:  47%|███████████▎            | 470/1000 [23:38<26:35,  3.01s/it]Dataset Size: 10%, Epoch: 471, Train Loss: 9.427084999084473, Val Loss: 10.178894179207939

Epoch Progress:  47%|███████████▎            | 471/1000 [23:41<26:33,  3.01s/it]
Epoch Progress:  47%|███████████▎            | 472/1000 [23:44<26:29,  3.01s/it]
Epoch Progress:  47%|███████████▎            | 473/1000 [23:47<26:26,  3.01s/it]
Epoch Progress:  47%|███████████▍            | 474/1000 [23:50<26:25,  3.01s/it]
Epoch Progress:  48%|███████████▍            | 475/1000 [23:53<26:23,  3.02s/it]
Epoch Progress:  48%|███████████▍            | 476/1000 [23:56<26:20,  3.02s/it]
Epoch Progress:  48%|███████████▍            | 477/1000 [23:59<26:16,  3.02s/it]
Epoch Progress:  48%|███████████▍            | 478/1000 [24:02<26:13,  3.01s/it]
Epoch Progress:  48%|███████████▍            | 479/1000 [24:05<26:09,  3.01s/it]
Epoch Progress:  48%|███████████▌            | 480/1000 [24:08<26:06,  3.01s/it]Dataset Size: 10%, Epoch: 481, Train Loss: 9.421845347086588, Val Loss: 10.185214327527332

Epoch Progress:  48%|███████████▌            | 481/1000 [24:11<26:04,  3.01s/it]
Epoch Progress:  48%|███████████▌            | 482/1000 [24:14<26:00,  3.01s/it]
Epoch Progress:  48%|███████████▌            | 483/1000 [24:17<25:56,  3.01s/it]
Epoch Progress:  48%|███████████▌            | 484/1000 [24:20<25:56,  3.02s/it]
Epoch Progress:  48%|███████████▋            | 485/1000 [24:23<25:54,  3.02s/it]
Epoch Progress:  49%|███████████▋            | 486/1000 [24:26<25:49,  3.01s/it]
Epoch Progress:  49%|███████████▋            | 487/1000 [24:29<25:44,  3.01s/it]
Epoch Progress:  49%|███████████▋            | 488/1000 [24:32<25:42,  3.01s/it]
Epoch Progress:  49%|███████████▋            | 489/1000 [24:35<25:38,  3.01s/it]
Epoch Progress:  49%|███████████▊            | 490/1000 [24:38<25:35,  3.01s/it]Dataset Size: 10%, Epoch: 491, Train Loss: 9.420332768758138, Val Loss: 10.183459281921387

Epoch Progress:  49%|███████████▊            | 491/1000 [24:41<25:33,  3.01s/it]
Epoch Progress:  49%|███████████▊            | 492/1000 [24:44<25:30,  3.01s/it]
Epoch Progress:  49%|███████████▊            | 493/1000 [24:47<25:27,  3.01s/it]
Epoch Progress:  49%|███████████▊            | 494/1000 [24:51<25:23,  3.01s/it]
Epoch Progress:  50%|███████████▉            | 495/1000 [24:54<25:23,  3.02s/it]
Epoch Progress:  50%|███████████▉            | 496/1000 [24:57<25:19,  3.01s/it]
Epoch Progress:  50%|███████████▉            | 497/1000 [25:00<25:15,  3.01s/it]
Epoch Progress:  50%|███████████▉            | 498/1000 [25:03<25:13,  3.01s/it]
Epoch Progress:  50%|███████████▉            | 499/1000 [25:06<25:09,  3.01s/it]
Epoch Progress:  50%|████████████            | 500/1000 [25:09<25:07,  3.01s/it]Dataset Size: 10%, Epoch: 501, Train Loss: 9.41591204325358, Val Loss: 10.184662162483512

Epoch Progress:  50%|████████████            | 501/1000 [25:12<25:04,  3.02s/it]
Epoch Progress:  50%|████████████            | 502/1000 [25:15<25:00,  3.01s/it]
Epoch Progress:  50%|████████████            | 503/1000 [25:18<24:56,  3.01s/it]
Epoch Progress:  50%|████████████            | 504/1000 [25:21<24:55,  3.02s/it]
Epoch Progress:  50%|████████████            | 505/1000 [25:24<24:53,  3.02s/it]
Epoch Progress:  51%|████████████▏           | 506/1000 [25:27<24:49,  3.02s/it]
Epoch Progress:  51%|████████████▏           | 507/1000 [25:30<24:46,  3.01s/it]
Epoch Progress:  51%|████████████▏           | 508/1000 [25:33<24:43,  3.01s/it]
Epoch Progress:  51%|████████████▏           | 509/1000 [25:36<24:38,  3.01s/it]
Epoch Progress:  51%|████████████▏           | 510/1000 [25:39<24:36,  3.01s/it]Dataset Size: 10%, Epoch: 511, Train Loss: 9.415752919514974, Val Loss: 10.184651250963087

Epoch Progress:  51%|████████████▎           | 511/1000 [25:42<24:33,  3.01s/it]
Epoch Progress:  51%|████████████▎           | 512/1000 [25:45<24:30,  3.01s/it]
Epoch Progress:  51%|████████████▎           | 513/1000 [25:48<24:26,  3.01s/it]
Epoch Progress:  51%|████████████▎           | 514/1000 [25:51<24:23,  3.01s/it]
Epoch Progress:  52%|████████████▎           | 515/1000 [25:54<24:20,  3.01s/it]
Epoch Progress:  52%|████████████▍           | 516/1000 [25:57<24:19,  3.02s/it]
Epoch Progress:  52%|████████████▍           | 517/1000 [26:00<24:16,  3.02s/it]
Epoch Progress:  52%|████████████▍           | 518/1000 [26:03<24:12,  3.01s/it]
Epoch Progress:  52%|████████████▍           | 519/1000 [26:06<24:09,  3.01s/it]
Epoch Progress:  52%|████████████▍           | 520/1000 [26:09<24:06,  3.01s/it]Dataset Size: 10%, Epoch: 521, Train Loss: 9.415646336873372, Val Loss: 10.182012718993349

Epoch Progress:  52%|████████████▌           | 521/1000 [26:12<24:04,  3.01s/it]
Epoch Progress:  52%|████████████▌           | 522/1000 [26:15<24:00,  3.01s/it]
Epoch Progress:  52%|████████████▌           | 523/1000 [26:18<23:57,  3.01s/it]
Epoch Progress:  52%|████████████▌           | 524/1000 [26:21<23:55,  3.02s/it]
Epoch Progress:  52%|████████████▌           | 525/1000 [26:24<23:52,  3.02s/it]
Epoch Progress:  53%|████████████▌           | 526/1000 [26:27<23:50,  3.02s/it]
Epoch Progress:  53%|████████████▋           | 527/1000 [26:30<23:47,  3.02s/it]
Epoch Progress:  53%|████████████▋           | 528/1000 [26:33<23:44,  3.02s/it]
Epoch Progress:  53%|████████████▋           | 529/1000 [26:36<23:39,  3.01s/it]
Epoch Progress:  53%|████████████▋           | 530/1000 [26:39<23:36,  3.01s/it]Dataset Size: 10%, Epoch: 531, Train Loss: 9.414870986938476, Val Loss: 10.184135672333953

Epoch Progress:  53%|████████████▋           | 531/1000 [26:42<23:34,  3.02s/it]
Epoch Progress:  53%|████████████▊           | 532/1000 [26:45<23:31,  3.02s/it]
Epoch Progress:  53%|████████████▊           | 533/1000 [26:48<23:26,  3.01s/it]
Epoch Progress:  53%|████████████▊           | 534/1000 [26:51<23:24,  3.01s/it]
Epoch Progress:  54%|████████████▊           | 535/1000 [26:54<23:19,  3.01s/it]
Epoch Progress:  54%|████████████▊           | 536/1000 [26:57<23:16,  3.01s/it]
Epoch Progress:  54%|████████████▉           | 537/1000 [27:00<23:16,  3.02s/it]
Epoch Progress:  54%|████████████▉           | 538/1000 [27:03<23:12,  3.01s/it]
Epoch Progress:  54%|████████████▉           | 539/1000 [27:06<23:07,  3.01s/it]
Epoch Progress:  54%|████████████▉           | 540/1000 [27:09<23:03,  3.01s/it]Dataset Size: 10%, Epoch: 541, Train Loss: 9.41262430826823, Val Loss: 10.181039822566046

Epoch Progress:  54%|████████████▉           | 541/1000 [27:12<23:02,  3.01s/it]
Epoch Progress:  54%|█████████████           | 542/1000 [27:15<22:58,  3.01s/it]
Epoch Progress:  54%|█████████████           | 543/1000 [27:18<22:56,  3.01s/it]
Epoch Progress:  54%|█████████████           | 544/1000 [27:21<22:54,  3.01s/it]
Epoch Progress:  55%|█████████████           | 545/1000 [27:24<22:50,  3.01s/it]
Epoch Progress:  55%|█████████████           | 546/1000 [27:27<22:46,  3.01s/it]
Epoch Progress:  55%|█████████████▏          | 547/1000 [27:30<22:44,  3.01s/it]
Epoch Progress:  55%|█████████████▏          | 548/1000 [27:33<22:44,  3.02s/it]
Epoch Progress:  55%|█████████████▏          | 549/1000 [27:36<22:40,  3.02s/it]
Epoch Progress:  55%|█████████████▏          | 550/1000 [27:39<22:37,  3.02s/it]Dataset Size: 10%, Epoch: 551, Train Loss: 9.412384859720866, Val Loss: 10.182228695262562

Epoch Progress:  55%|█████████████▏          | 551/1000 [27:42<22:34,  3.02s/it]
Epoch Progress:  55%|█████████████▏          | 552/1000 [27:45<22:29,  3.01s/it]
Epoch Progress:  55%|█████████████▎          | 553/1000 [27:48<22:25,  3.01s/it]
Epoch Progress:  55%|█████████████▎          | 554/1000 [27:51<22:24,  3.01s/it]
Epoch Progress:  56%|█████████████▎          | 555/1000 [27:54<22:20,  3.01s/it]
Epoch Progress:  56%|█████████████▎          | 556/1000 [27:57<22:17,  3.01s/it]
Epoch Progress:  56%|█████████████▎          | 557/1000 [28:00<22:14,  3.01s/it]
Epoch Progress:  56%|█████████████▍          | 558/1000 [28:03<22:13,  3.02s/it]
Epoch Progress:  56%|█████████████▍          | 559/1000 [28:06<22:09,  3.02s/it]
Epoch Progress:  56%|█████████████▍          | 560/1000 [28:09<22:05,  3.01s/it]Dataset Size: 10%, Epoch: 561, Train Loss: 9.408275438944498, Val Loss: 10.179993988631608

Epoch Progress:  56%|█████████████▍          | 561/1000 [28:12<22:03,  3.01s/it]
Epoch Progress:  56%|█████████████▍          | 562/1000 [28:15<21:59,  3.01s/it]
Epoch Progress:  56%|█████████████▌          | 563/1000 [28:18<21:55,  3.01s/it]
Epoch Progress:  56%|█████████████▌          | 564/1000 [28:21<21:54,  3.01s/it]
Epoch Progress:  56%|█████████████▌          | 565/1000 [28:24<21:50,  3.01s/it]
Epoch Progress:  57%|█████████████▌          | 566/1000 [28:28<21:47,  3.01s/it]
Epoch Progress:  57%|█████████████▌          | 567/1000 [28:31<21:43,  3.01s/it]
Epoch Progress:  57%|█████████████▋          | 568/1000 [28:34<21:42,  3.02s/it]
Epoch Progress:  57%|█████████████▋          | 569/1000 [28:37<21:39,  3.02s/it]
Epoch Progress:  57%|█████████████▋          | 570/1000 [28:40<21:35,  3.01s/it]Dataset Size: 10%, Epoch: 571, Train Loss: 9.409005800882975, Val Loss: 10.1794403373421

Epoch Progress:  57%|█████████████▋          | 571/1000 [28:43<21:33,  3.02s/it]
Epoch Progress:  57%|█████████████▋          | 572/1000 [28:46<21:29,  3.01s/it]
Epoch Progress:  57%|█████████████▊          | 573/1000 [28:49<21:26,  3.01s/it]
Epoch Progress:  57%|█████████████▊          | 574/1000 [28:52<21:23,  3.01s/it]
Epoch Progress:  57%|█████████████▊          | 575/1000 [28:55<21:19,  3.01s/it]
Epoch Progress:  58%|█████████████▊          | 576/1000 [28:58<21:17,  3.01s/it]
Epoch Progress:  58%|█████████████▊          | 577/1000 [29:01<21:14,  3.01s/it]
Epoch Progress:  58%|█████████████▊          | 578/1000 [29:04<21:11,  3.01s/it]
Epoch Progress:  58%|█████████████▉          | 579/1000 [29:07<21:09,  3.01s/it]
Epoch Progress:  58%|█████████████▉          | 580/1000 [29:10<21:06,  3.01s/it]Dataset Size: 10%, Epoch: 581, Train Loss: 9.40619561513265, Val Loss: 10.180375916617256

Epoch Progress:  58%|█████████████▉          | 581/1000 [29:13<21:03,  3.01s/it]
Epoch Progress:  58%|█████████████▉          | 582/1000 [29:16<20:59,  3.01s/it]
Epoch Progress:  58%|█████████████▉          | 583/1000 [29:19<20:56,  3.01s/it]
Epoch Progress:  58%|██████████████          | 584/1000 [29:22<20:54,  3.02s/it]
Epoch Progress:  58%|██████████████          | 585/1000 [29:25<20:51,  3.01s/it]
Epoch Progress:  59%|██████████████          | 586/1000 [29:28<20:46,  3.01s/it]
Epoch Progress:  59%|██████████████          | 587/1000 [29:31<20:45,  3.02s/it]
Epoch Progress:  59%|██████████████          | 588/1000 [29:34<20:41,  3.01s/it]
Epoch Progress:  59%|██████████████▏         | 589/1000 [29:37<20:38,  3.01s/it]
Epoch Progress:  59%|██████████████▏         | 590/1000 [29:40<20:36,  3.02s/it]Dataset Size: 10%, Epoch: 591, Train Loss: 9.404383544921876, Val Loss: 10.183152112093838

Epoch Progress:  59%|██████████████▏         | 591/1000 [29:43<20:33,  3.02s/it]
Epoch Progress:  59%|██████████████▏         | 592/1000 [29:46<20:30,  3.02s/it]
Epoch Progress:  59%|██████████████▏         | 593/1000 [29:49<20:26,  3.01s/it]
Epoch Progress:  59%|██████████████▎         | 594/1000 [29:52<20:23,  3.01s/it]
Epoch Progress:  60%|██████████████▎         | 595/1000 [29:55<20:20,  3.01s/it]
Epoch Progress:  60%|██████████████▎         | 596/1000 [29:58<20:17,  3.01s/it]
Epoch Progress:  60%|██████████████▎         | 597/1000 [30:01<20:14,  3.01s/it]
Epoch Progress:  60%|██████████████▎         | 598/1000 [30:04<20:11,  3.01s/it]
Epoch Progress:  60%|██████████████▍         | 599/1000 [30:07<20:07,  3.01s/it]
Epoch Progress:  60%|██████████████▍         | 600/1000 [30:10<20:05,  3.01s/it]Dataset Size: 10%, Epoch: 601, Train Loss: 9.40819839477539, Val Loss: 10.182763310221883

Epoch Progress:  60%|██████████████▍         | 601/1000 [30:13<20:03,  3.02s/it]
Epoch Progress:  60%|██████████████▍         | 602/1000 [30:16<19:58,  3.01s/it]
Epoch Progress:  60%|██████████████▍         | 603/1000 [30:19<19:54,  3.01s/it]
Epoch Progress:  60%|██████████████▍         | 604/1000 [30:22<19:52,  3.01s/it]
Epoch Progress:  60%|██████████████▌         | 605/1000 [30:25<19:49,  3.01s/it]
Epoch Progress:  61%|██████████████▌         | 606/1000 [30:28<19:46,  3.01s/it]
Epoch Progress:  61%|██████████████▌         | 607/1000 [30:31<19:44,  3.02s/it]
Epoch Progress:  61%|██████████████▌         | 608/1000 [30:34<19:41,  3.01s/it]
Epoch Progress:  61%|██████████████▌         | 609/1000 [30:37<19:38,  3.01s/it]
Epoch Progress:  61%|██████████████▋         | 610/1000 [30:40<19:38,  3.02s/it]Dataset Size: 10%, Epoch: 611, Train Loss: 9.404651590983073, Val Loss: 10.18354692087545

Epoch Progress:  61%|██████████████▋         | 611/1000 [30:43<19:36,  3.03s/it]
Epoch Progress:  61%|██████████████▋         | 612/1000 [30:46<19:33,  3.02s/it]
Epoch Progress:  61%|██████████████▋         | 613/1000 [30:49<19:28,  3.02s/it]
Epoch Progress:  61%|██████████████▋         | 614/1000 [30:52<19:24,  3.02s/it]
Epoch Progress:  62%|██████████████▊         | 615/1000 [30:55<19:20,  3.02s/it]
Epoch Progress:  62%|██████████████▊         | 616/1000 [30:58<19:16,  3.01s/it]
Epoch Progress:  62%|██████████████▊         | 617/1000 [31:01<19:14,  3.01s/it]
Epoch Progress:  62%|██████████████▊         | 618/1000 [31:04<19:11,  3.01s/it]
Epoch Progress:  62%|██████████████▊         | 619/1000 [31:07<19:08,  3.01s/it]
Epoch Progress:  62%|██████████████▉         | 620/1000 [31:10<19:05,  3.01s/it]Dataset Size: 10%, Epoch: 621, Train Loss: 9.399875399271647, Val Loss: 10.183809788196118

Epoch Progress:  62%|██████████████▉         | 621/1000 [31:13<19:04,  3.02s/it]
Epoch Progress:  62%|██████████████▉         | 622/1000 [31:16<18:59,  3.02s/it]
Epoch Progress:  62%|██████████████▉         | 623/1000 [31:19<18:56,  3.01s/it]
Epoch Progress:  62%|██████████████▉         | 624/1000 [31:22<18:54,  3.02s/it]
Epoch Progress:  62%|███████████████         | 625/1000 [31:25<18:53,  3.02s/it]
Epoch Progress:  63%|███████████████         | 626/1000 [31:28<18:48,  3.02s/it]
Epoch Progress:  63%|███████████████         | 627/1000 [31:31<18:45,  3.02s/it]
Epoch Progress:  63%|███████████████         | 628/1000 [31:34<18:41,  3.01s/it]
Epoch Progress:  63%|███████████████         | 629/1000 [31:37<18:39,  3.02s/it]
Epoch Progress:  63%|███████████████         | 630/1000 [31:40<18:36,  3.02s/it]Dataset Size: 10%, Epoch: 631, Train Loss: 9.405898628234864, Val Loss: 10.18509698843027

Epoch Progress:  63%|███████████████▏        | 631/1000 [31:43<18:34,  3.02s/it]
Epoch Progress:  63%|███████████████▏        | 632/1000 [31:47<18:32,  3.02s/it]
Epoch Progress:  63%|███████████████▏        | 633/1000 [31:50<18:28,  3.02s/it]
Epoch Progress:  63%|███████████████▏        | 634/1000 [31:53<18:26,  3.02s/it]
Epoch Progress:  64%|███████████████▏        | 635/1000 [31:56<18:21,  3.02s/it]
Epoch Progress:  64%|███████████████▎        | 636/1000 [31:59<18:17,  3.02s/it]
Epoch Progress:  64%|███████████████▎        | 637/1000 [32:02<18:14,  3.01s/it]
Epoch Progress:  64%|███████████████▎        | 638/1000 [32:05<18:10,  3.01s/it]
Epoch Progress:  64%|███████████████▎        | 639/1000 [32:08<18:07,  3.01s/it]
Epoch Progress:  64%|███████████████▎        | 640/1000 [32:11<18:06,  3.02s/it]Dataset Size: 10%, Epoch: 641, Train Loss: 9.400863812764486, Val Loss: 10.182668475361613

Epoch Progress:  64%|███████████████▍        | 641/1000 [32:14<18:03,  3.02s/it]
Epoch Progress:  64%|███████████████▍        | 642/1000 [32:17<18:02,  3.02s/it]
Epoch Progress:  64%|███████████████▍        | 643/1000 [32:20<17:58,  3.02s/it]
Epoch Progress:  64%|███████████████▍        | 644/1000 [32:23<17:54,  3.02s/it]
Epoch Progress:  64%|███████████████▍        | 645/1000 [32:26<17:51,  3.02s/it]
Epoch Progress:  65%|███████████████▌        | 646/1000 [32:29<17:47,  3.02s/it]
Epoch Progress:  65%|███████████████▌        | 647/1000 [32:32<17:43,  3.01s/it]
Epoch Progress:  65%|███████████████▌        | 648/1000 [32:35<17:40,  3.01s/it]
Epoch Progress:  65%|███████████████▌        | 649/1000 [32:38<17:36,  3.01s/it]
Epoch Progress:  65%|███████████████▌        | 650/1000 [32:41<17:33,  3.01s/it]Dataset Size: 10%, Epoch: 651, Train Loss: 9.403468653361003, Val Loss: 10.183950572818905

Epoch Progress:  65%|███████████████▌        | 651/1000 [32:44<17:31,  3.01s/it]
Epoch Progress:  65%|███████████████▋        | 652/1000 [32:47<17:28,  3.01s/it]
Epoch Progress:  65%|███████████████▋        | 653/1000 [32:50<17:27,  3.02s/it]
Epoch Progress:  65%|███████████████▋        | 654/1000 [32:53<17:24,  3.02s/it]
Epoch Progress:  66%|███████████████▋        | 655/1000 [32:56<17:21,  3.02s/it]
Epoch Progress:  66%|███████████████▋        | 656/1000 [32:59<17:17,  3.02s/it]
Epoch Progress:  66%|███████████████▊        | 657/1000 [33:02<17:14,  3.02s/it]
Epoch Progress:  66%|███████████████▊        | 658/1000 [33:05<17:11,  3.02s/it]
Epoch Progress:  66%|███████████████▊        | 659/1000 [33:08<17:08,  3.02s/it]
Epoch Progress:  66%|███████████████▊        | 660/1000 [33:11<17:06,  3.02s/it]Dataset Size: 10%, Epoch: 661, Train Loss: 9.400084139506022, Val Loss: 10.183144234991692

Epoch Progress:  66%|███████████████▊        | 661/1000 [33:14<17:02,  3.02s/it]
Epoch Progress:  66%|███████████████▉        | 662/1000 [33:17<16:59,  3.02s/it]
Epoch Progress:  66%|███████████████▉        | 663/1000 [33:20<16:57,  3.02s/it]
Epoch Progress:  66%|███████████████▉        | 664/1000 [33:23<16:54,  3.02s/it]
Epoch Progress:  66%|███████████████▉        | 665/1000 [33:26<16:50,  3.02s/it]
Epoch Progress:  67%|███████████████▉        | 666/1000 [33:29<16:46,  3.01s/it]
Epoch Progress:  67%|████████████████        | 667/1000 [33:32<16:43,  3.01s/it]
Epoch Progress:  67%|████████████████        | 668/1000 [33:35<16:39,  3.01s/it]
Epoch Progress:  67%|████████████████        | 669/1000 [33:38<16:36,  3.01s/it]
Epoch Progress:  67%|████████████████        | 670/1000 [33:41<16:34,  3.01s/it]Dataset Size: 10%, Epoch: 671, Train Loss: 9.39995703379313, Val Loss: 10.184636326579305

Epoch Progress:  67%|████████████████        | 671/1000 [33:44<16:30,  3.01s/it]
Epoch Progress:  67%|████████████████▏       | 672/1000 [33:47<16:26,  3.01s/it]
Epoch Progress:  67%|████████████████▏       | 673/1000 [33:50<16:23,  3.01s/it]
Epoch Progress:  67%|████████████████▏       | 674/1000 [33:53<16:22,  3.02s/it]
Epoch Progress:  68%|████████████████▏       | 675/1000 [33:56<16:19,  3.01s/it]
Epoch Progress:  68%|████████████████▏       | 676/1000 [33:59<16:15,  3.01s/it]
Epoch Progress:  68%|████████████████▏       | 677/1000 [34:02<16:12,  3.01s/it]
Epoch Progress:  68%|████████████████▎       | 678/1000 [34:05<16:08,  3.01s/it]
Epoch Progress:  68%|████████████████▎       | 679/1000 [34:08<16:06,  3.01s/it]
Epoch Progress:  68%|████████████████▎       | 680/1000 [34:11<16:05,  3.02s/it]Dataset Size: 10%, Epoch: 681, Train Loss: 9.400485699971517, Val Loss: 10.18371193130295

Epoch Progress:  68%|████████████████▎       | 681/1000 [34:14<16:00,  3.01s/it]
Epoch Progress:  68%|████████████████▎       | 682/1000 [34:17<15:57,  3.01s/it]
Epoch Progress:  68%|████████████████▍       | 683/1000 [34:20<15:54,  3.01s/it]
Epoch Progress:  68%|████████████████▍       | 684/1000 [34:23<15:53,  3.02s/it]
Epoch Progress:  68%|████████████████▍       | 685/1000 [34:26<15:51,  3.02s/it]
Epoch Progress:  69%|████████████████▍       | 686/1000 [34:29<15:47,  3.02s/it]
Epoch Progress:  69%|████████████████▍       | 687/1000 [34:32<15:44,  3.02s/it]
Epoch Progress:  69%|████████████████▌       | 688/1000 [34:35<15:39,  3.01s/it]
Epoch Progress:  69%|████████████████▌       | 689/1000 [34:38<15:36,  3.01s/it]
Epoch Progress:  69%|████████████████▌       | 690/1000 [34:41<15:34,  3.01s/it]Dataset Size: 10%, Epoch: 691, Train Loss: 9.398122088114421, Val Loss: 10.18572845706692

Epoch Progress:  69%|████████████████▌       | 691/1000 [34:44<15:31,  3.01s/it]
Epoch Progress:  69%|████████████████▌       | 692/1000 [34:47<15:27,  3.01s/it]
Epoch Progress:  69%|████████████████▋       | 693/1000 [34:50<15:24,  3.01s/it]
Epoch Progress:  69%|████████████████▋       | 694/1000 [34:53<15:22,  3.01s/it]
Epoch Progress:  70%|████████████████▋       | 695/1000 [34:56<15:19,  3.02s/it]
Epoch Progress:  70%|████████████████▋       | 696/1000 [34:59<15:15,  3.01s/it]
Epoch Progress:  70%|████████████████▋       | 697/1000 [35:02<15:12,  3.01s/it]
Epoch Progress:  70%|████████████████▊       | 698/1000 [35:05<15:10,  3.01s/it]
Epoch Progress:  70%|████████████████▊       | 699/1000 [35:08<15:06,  3.01s/it]
Epoch Progress:  70%|████████████████▊       | 700/1000 [35:12<15:04,  3.01s/it]Dataset Size: 10%, Epoch: 701, Train Loss: 9.39737637837728, Val Loss: 10.185459818158831

Epoch Progress:  70%|████████████████▊       | 701/1000 [35:15<15:00,  3.01s/it]
Epoch Progress:  70%|████████████████▊       | 702/1000 [35:18<14:57,  3.01s/it]
Epoch Progress:  70%|████████████████▊       | 703/1000 [35:21<14:54,  3.01s/it]
Epoch Progress:  70%|████████████████▉       | 704/1000 [35:24<14:51,  3.01s/it]
Epoch Progress:  70%|████████████████▉       | 705/1000 [35:27<14:52,  3.02s/it]
Epoch Progress:  71%|████████████████▉       | 706/1000 [35:30<14:48,  3.02s/it]
Epoch Progress:  71%|████████████████▉       | 707/1000 [35:33<14:45,  3.02s/it]
Epoch Progress:  71%|████████████████▉       | 708/1000 [35:36<14:40,  3.02s/it]
Epoch Progress:  71%|█████████████████       | 709/1000 [35:39<14:36,  3.01s/it]
Epoch Progress:  71%|█████████████████       | 710/1000 [35:42<14:32,  3.01s/it]Dataset Size: 10%, Epoch: 711, Train Loss: 9.400756581624348, Val Loss: 10.184616026940283

Epoch Progress:  71%|█████████████████       | 711/1000 [35:45<14:29,  3.01s/it]
Epoch Progress:  71%|█████████████████       | 712/1000 [35:48<14:26,  3.01s/it]
Epoch Progress:  71%|█████████████████       | 713/1000 [35:51<14:23,  3.01s/it]
Epoch Progress:  71%|█████████████████▏      | 714/1000 [35:54<14:20,  3.01s/it]
Epoch Progress:  72%|█████████████████▏      | 715/1000 [35:57<14:17,  3.01s/it]
Epoch Progress:  72%|█████████████████▏      | 716/1000 [36:00<14:15,  3.01s/it]
Epoch Progress:  72%|█████████████████▏      | 717/1000 [36:03<14:13,  3.01s/it]
Epoch Progress:  72%|█████████████████▏      | 718/1000 [36:06<14:08,  3.01s/it]
Epoch Progress:  72%|█████████████████▎      | 719/1000 [36:09<14:05,  3.01s/it]
Epoch Progress:  72%|█████████████████▎      | 720/1000 [36:12<14:03,  3.01s/it]Dataset Size: 10%, Epoch: 721, Train Loss: 9.39364751180013, Val Loss: 10.188064426570744

Epoch Progress:  72%|█████████████████▎      | 721/1000 [36:15<13:59,  3.01s/it]
Epoch Progress:  72%|█████████████████▎      | 722/1000 [36:18<13:56,  3.01s/it]
Epoch Progress:  72%|█████████████████▎      | 723/1000 [36:21<13:54,  3.01s/it]
Epoch Progress:  72%|█████████████████▍      | 724/1000 [36:24<13:51,  3.01s/it]
Epoch Progress:  72%|█████████████████▍      | 725/1000 [36:27<13:47,  3.01s/it]
Epoch Progress:  73%|█████████████████▍      | 726/1000 [36:30<13:45,  3.01s/it]
Epoch Progress:  73%|█████████████████▍      | 727/1000 [36:33<13:43,  3.01s/it]
Epoch Progress:  73%|█████████████████▍      | 728/1000 [36:36<13:39,  3.01s/it]
Epoch Progress:  73%|█████████████████▍      | 729/1000 [36:39<13:36,  3.01s/it]
Epoch Progress:  73%|█████████████████▌      | 730/1000 [36:42<13:32,  3.01s/it]Dataset Size: 10%, Epoch: 731, Train Loss: 9.397931226094563, Val Loss: 10.188692390144645

Epoch Progress:  73%|█████████████████▌      | 731/1000 [36:45<13:29,  3.01s/it]
Epoch Progress:  73%|█████████████████▌      | 732/1000 [36:48<13:27,  3.01s/it]
Epoch Progress:  73%|█████████████████▌      | 733/1000 [36:51<13:25,  3.02s/it]
Epoch Progress:  73%|█████████████████▌      | 734/1000 [36:54<13:21,  3.01s/it]
Epoch Progress:  74%|█████████████████▋      | 735/1000 [36:57<13:18,  3.01s/it]
Epoch Progress:  74%|█████████████████▋      | 736/1000 [37:00<13:15,  3.01s/it]
Epoch Progress:  74%|█████████████████▋      | 737/1000 [37:03<13:13,  3.02s/it]
Epoch Progress:  74%|█████████████████▋      | 738/1000 [37:06<13:10,  3.02s/it]
Epoch Progress:  74%|█████████████████▋      | 739/1000 [37:09<13:07,  3.02s/it]
Epoch Progress:  74%|█████████████████▊      | 740/1000 [37:12<13:04,  3.02s/it]Dataset Size: 10%, Epoch: 741, Train Loss: 9.394523811340331, Val Loss: 10.18710065816904

Epoch Progress:  74%|█████████████████▊      | 741/1000 [37:15<13:00,  3.02s/it]
Epoch Progress:  74%|█████████████████▊      | 742/1000 [37:18<12:57,  3.01s/it]
Epoch Progress:  74%|█████████████████▊      | 743/1000 [37:21<12:55,  3.02s/it]
Epoch Progress:  74%|█████████████████▊      | 744/1000 [37:24<12:52,  3.02s/it]
Epoch Progress:  74%|█████████████████▉      | 745/1000 [37:27<12:48,  3.01s/it]
Epoch Progress:  75%|█████████████████▉      | 746/1000 [37:30<12:44,  3.01s/it]
Epoch Progress:  75%|█████████████████▉      | 747/1000 [37:33<12:44,  3.02s/it]
Epoch Progress:  75%|█████████████████▉      | 748/1000 [37:36<12:40,  3.02s/it]
Epoch Progress:  75%|█████████████████▉      | 749/1000 [37:39<12:36,  3.01s/it]
Epoch Progress:  75%|██████████████████      | 750/1000 [37:42<12:33,  3.01s/it]Dataset Size: 10%, Epoch: 751, Train Loss: 9.398911603291829, Val Loss: 10.189544008923816

Epoch Progress:  75%|██████████████████      | 751/1000 [37:45<12:30,  3.01s/it]
Epoch Progress:  75%|██████████████████      | 752/1000 [37:48<12:26,  3.01s/it]
Epoch Progress:  75%|██████████████████      | 753/1000 [37:51<12:23,  3.01s/it]
Epoch Progress:  75%|██████████████████      | 754/1000 [37:54<12:20,  3.01s/it]
Epoch Progress:  76%|██████████████████      | 755/1000 [37:57<12:17,  3.01s/it]
Epoch Progress:  76%|██████████████████▏     | 756/1000 [38:00<12:13,  3.01s/it]
Epoch Progress:  76%|██████████████████▏     | 757/1000 [38:03<12:11,  3.01s/it]
Epoch Progress:  76%|██████████████████▏     | 758/1000 [38:06<12:09,  3.01s/it]
Epoch Progress:  76%|██████████████████▏     | 759/1000 [38:09<12:06,  3.01s/it]
Epoch Progress:  76%|██████████████████▏     | 760/1000 [38:12<12:03,  3.02s/it]Dataset Size: 10%, Epoch: 761, Train Loss: 9.39507630666097, Val Loss: 10.188375002377994

Epoch Progress:  76%|██████████████████▎     | 761/1000 [38:15<12:00,  3.01s/it]
Epoch Progress:  76%|██████████████████▎     | 762/1000 [38:18<11:56,  3.01s/it]
Epoch Progress:  76%|██████████████████▎     | 763/1000 [38:21<11:54,  3.01s/it]
Epoch Progress:  76%|██████████████████▎     | 764/1000 [38:24<11:50,  3.01s/it]
Epoch Progress:  76%|██████████████████▎     | 765/1000 [38:27<11:47,  3.01s/it]
Epoch Progress:  77%|██████████████████▍     | 766/1000 [38:30<11:45,  3.01s/it]
Epoch Progress:  77%|██████████████████▍     | 767/1000 [38:33<11:42,  3.01s/it]
Epoch Progress:  77%|██████████████████▍     | 768/1000 [38:36<11:40,  3.02s/it]
Epoch Progress:  77%|██████████████████▍     | 769/1000 [38:39<11:36,  3.02s/it]
Epoch Progress:  77%|██████████████████▍     | 770/1000 [38:42<11:33,  3.02s/it]Dataset Size: 10%, Epoch: 771, Train Loss: 9.39448434193929, Val Loss: 10.187133900530927

Epoch Progress:  77%|██████████████████▌     | 771/1000 [38:45<11:30,  3.01s/it]
Epoch Progress:  77%|██████████████████▌     | 772/1000 [38:48<11:26,  3.01s/it]
Epoch Progress:  77%|██████████████████▌     | 773/1000 [38:51<11:24,  3.01s/it]
Epoch Progress:  77%|██████████████████▌     | 774/1000 [38:54<11:21,  3.01s/it]
Epoch Progress:  78%|██████████████████▌     | 775/1000 [38:58<11:17,  3.01s/it]
Epoch Progress:  78%|██████████████████▌     | 776/1000 [39:01<11:13,  3.01s/it]
Epoch Progress:  78%|██████████████████▋     | 777/1000 [39:04<11:10,  3.01s/it]
Epoch Progress:  78%|██████████████████▋     | 778/1000 [39:07<11:07,  3.01s/it]
Epoch Progress:  78%|██████████████████▋     | 779/1000 [39:10<11:05,  3.01s/it]
Epoch Progress:  78%|██████████████████▋     | 780/1000 [39:13<11:02,  3.01s/it]Dataset Size: 10%, Epoch: 781, Train Loss: 9.39385882059733, Val Loss: 10.185090944364473

Epoch Progress:  78%|██████████████████▋     | 781/1000 [39:16<10:59,  3.01s/it]
Epoch Progress:  78%|██████████████████▊     | 782/1000 [39:19<10:56,  3.01s/it]
Epoch Progress:  78%|██████████████████▊     | 783/1000 [39:22<10:53,  3.01s/it]
Epoch Progress:  78%|██████████████████▊     | 784/1000 [39:25<10:50,  3.01s/it]
Epoch Progress:  78%|██████████████████▊     | 785/1000 [39:28<10:46,  3.01s/it]
Epoch Progress:  79%|██████████████████▊     | 786/1000 [39:31<10:44,  3.01s/it]
Epoch Progress:  79%|██████████████████▉     | 787/1000 [39:34<10:41,  3.01s/it]
Epoch Progress:  79%|██████████████████▉     | 788/1000 [39:37<10:38,  3.01s/it]
Epoch Progress:  79%|██████████████████▉     | 789/1000 [39:40<10:36,  3.02s/it]
Epoch Progress:  79%|██████████████████▉     | 790/1000 [39:43<10:33,  3.02s/it]Dataset Size: 10%, Epoch: 791, Train Loss: 9.394086011250813, Val Loss: 10.18456184089958

Epoch Progress:  79%|██████████████████▉     | 791/1000 [39:46<10:29,  3.01s/it]
Epoch Progress:  79%|███████████████████     | 792/1000 [39:49<10:25,  3.01s/it]
Epoch Progress:  79%|███████████████████     | 793/1000 [39:52<10:22,  3.01s/it]
Epoch Progress:  79%|███████████████████     | 794/1000 [39:55<10:19,  3.01s/it]
Epoch Progress:  80%|███████████████████     | 795/1000 [39:58<10:17,  3.01s/it]
Epoch Progress:  80%|███████████████████     | 796/1000 [40:01<10:14,  3.01s/it]
Epoch Progress:  80%|███████████████████▏    | 797/1000 [40:04<10:11,  3.01s/it]
Epoch Progress:  80%|███████████████████▏    | 798/1000 [40:07<10:07,  3.01s/it]
Epoch Progress:  80%|███████████████████▏    | 799/1000 [40:10<10:04,  3.01s/it]
Epoch Progress:  80%|███████████████████▏    | 800/1000 [40:13<10:02,  3.01s/it]Dataset Size: 10%, Epoch: 801, Train Loss: 9.396981671651204, Val Loss: 10.18863007929418

Epoch Progress:  80%|███████████████████▏    | 801/1000 [40:16<09:59,  3.01s/it]
Epoch Progress:  80%|███████████████████▏    | 802/1000 [40:19<09:56,  3.01s/it]
Epoch Progress:  80%|███████████████████▎    | 803/1000 [40:22<09:53,  3.01s/it]
Epoch Progress:  80%|███████████████████▎    | 804/1000 [40:25<09:50,  3.01s/it]
Epoch Progress:  80%|███████████████████▎    | 805/1000 [40:28<09:47,  3.01s/it]
Epoch Progress:  81%|███████████████████▎    | 806/1000 [40:31<09:44,  3.01s/it]
Epoch Progress:  81%|███████████████████▎    | 807/1000 [40:34<09:41,  3.01s/it]
Epoch Progress:  81%|███████████████████▍    | 808/1000 [40:37<09:38,  3.01s/it]
Epoch Progress:  81%|███████████████████▍    | 809/1000 [40:40<09:34,  3.01s/it]
Epoch Progress:  81%|███████████████████▍    | 810/1000 [40:43<09:33,  3.02s/it]Dataset Size: 10%, Epoch: 811, Train Loss: 9.395495783487956, Val Loss: 10.186130436983975

Epoch Progress:  81%|███████████████████▍    | 811/1000 [40:46<09:30,  3.02s/it]
Epoch Progress:  81%|███████████████████▍    | 812/1000 [40:49<09:26,  3.01s/it]
Epoch Progress:  81%|███████████████████▌    | 813/1000 [40:52<09:23,  3.01s/it]
Epoch Progress:  81%|███████████████████▌    | 814/1000 [40:55<09:20,  3.01s/it]
Epoch Progress:  82%|███████████████████▌    | 815/1000 [40:58<09:16,  3.01s/it]
Epoch Progress:  82%|███████████████████▌    | 816/1000 [41:01<09:14,  3.01s/it]
Epoch Progress:  82%|███████████████████▌    | 817/1000 [41:04<09:11,  3.01s/it]
Epoch Progress:  82%|███████████████████▋    | 818/1000 [41:07<09:08,  3.01s/it]
Epoch Progress:  82%|███████████████████▋    | 819/1000 [41:10<09:05,  3.01s/it]
Epoch Progress:  82%|███████████████████▋    | 820/1000 [41:13<09:03,  3.02s/it]Dataset Size: 10%, Epoch: 821, Train Loss: 9.396372261047363, Val Loss: 10.18572491484803

Epoch Progress:  82%|███████████████████▋    | 821/1000 [41:16<09:00,  3.02s/it]
Epoch Progress:  82%|███████████████████▋    | 822/1000 [41:19<08:56,  3.02s/it]
Epoch Progress:  82%|███████████████████▊    | 823/1000 [41:22<08:54,  3.02s/it]
Epoch Progress:  82%|███████████████████▊    | 824/1000 [41:25<08:50,  3.01s/it]
Epoch Progress:  82%|███████████████████▊    | 825/1000 [41:28<08:46,  3.01s/it]
Epoch Progress:  83%|███████████████████▊    | 826/1000 [41:31<08:43,  3.01s/it]
Epoch Progress:  83%|███████████████████▊    | 827/1000 [41:34<08:40,  3.01s/it]
Epoch Progress:  83%|███████████████████▊    | 828/1000 [41:37<08:37,  3.01s/it]
Epoch Progress:  83%|███████████████████▉    | 829/1000 [41:40<08:34,  3.01s/it]
Epoch Progress:  83%|███████████████████▉    | 830/1000 [41:43<08:31,  3.01s/it]Dataset Size: 10%, Epoch: 831, Train Loss: 9.393953755696614, Val Loss: 10.18450439750374

Epoch Progress:  83%|███████████████████▉    | 831/1000 [41:46<08:29,  3.01s/it]
Epoch Progress:  83%|███████████████████▉    | 832/1000 [41:49<08:26,  3.02s/it]
Epoch Progress:  83%|███████████████████▉    | 833/1000 [41:52<08:23,  3.02s/it]
Epoch Progress:  83%|████████████████████    | 834/1000 [41:55<08:20,  3.01s/it]
Epoch Progress:  84%|████████████████████    | 835/1000 [41:58<08:17,  3.01s/it]
Epoch Progress:  84%|████████████████████    | 836/1000 [42:01<08:14,  3.02s/it]
Epoch Progress:  84%|████████████████████    | 837/1000 [42:04<08:11,  3.02s/it]
Epoch Progress:  84%|████████████████████    | 838/1000 [42:07<08:08,  3.01s/it]
Epoch Progress:  84%|████████████████████▏   | 839/1000 [42:10<08:05,  3.02s/it]
Epoch Progress:  84%|████████████████████▏   | 840/1000 [42:13<08:02,  3.02s/it]Dataset Size: 10%, Epoch: 841, Train Loss: 9.39544662475586, Val Loss: 10.186375073024205

Epoch Progress:  84%|████████████████████▏   | 841/1000 [42:16<07:59,  3.02s/it]
Epoch Progress:  84%|████████████████████▏   | 842/1000 [42:19<07:56,  3.02s/it]
Epoch Progress:  84%|████████████████████▏   | 843/1000 [42:22<07:53,  3.02s/it]
Epoch Progress:  84%|████████████████████▎   | 844/1000 [42:25<07:50,  3.02s/it]
Epoch Progress:  84%|████████████████████▎   | 845/1000 [42:28<07:46,  3.01s/it]
Epoch Progress:  85%|████████████████████▎   | 846/1000 [42:31<07:44,  3.01s/it]
Epoch Progress:  85%|████████████████████▎   | 847/1000 [42:34<07:40,  3.01s/it]
Epoch Progress:  85%|████████████████████▎   | 848/1000 [42:37<07:37,  3.01s/it]
Epoch Progress:  85%|████████████████████▍   | 849/1000 [42:40<07:34,  3.01s/it]
Epoch Progress:  85%|████████████████████▍   | 850/1000 [42:43<07:31,  3.01s/it]Dataset Size: 10%, Epoch: 851, Train Loss: 9.389985326131185, Val Loss: 10.187704643645844

Epoch Progress:  85%|████████████████████▍   | 851/1000 [42:46<07:28,  3.01s/it]
Epoch Progress:  85%|████████████████████▍   | 852/1000 [42:49<07:25,  3.01s/it]
Epoch Progress:  85%|████████████████████▍   | 853/1000 [42:53<07:23,  3.02s/it]
Epoch Progress:  85%|████████████████████▍   | 854/1000 [42:56<07:20,  3.02s/it]
Epoch Progress:  86%|████████████████████▌   | 855/1000 [42:59<07:17,  3.01s/it]
Epoch Progress:  86%|████████████████████▌   | 856/1000 [43:02<07:14,  3.01s/it]
Epoch Progress:  86%|████████████████████▌   | 857/1000 [43:05<07:10,  3.01s/it]
Epoch Progress:  86%|████████████████████▌   | 858/1000 [43:08<07:07,  3.01s/it]
Epoch Progress:  86%|████████████████████▌   | 859/1000 [43:11<07:05,  3.02s/it]
Epoch Progress:  86%|████████████████████▋   | 860/1000 [43:14<07:01,  3.01s/it]Dataset Size: 10%, Epoch: 861, Train Loss: 9.393795445760091, Val Loss: 10.188497431866534

Epoch Progress:  86%|████████████████████▋   | 861/1000 [43:17<06:58,  3.01s/it]
Epoch Progress:  86%|████████████████████▋   | 862/1000 [43:20<06:55,  3.01s/it]
Epoch Progress:  86%|████████████████████▋   | 863/1000 [43:23<06:53,  3.02s/it]
Epoch Progress:  86%|████████████████████▋   | 864/1000 [43:26<06:49,  3.01s/it]
Epoch Progress:  86%|████████████████████▊   | 865/1000 [43:29<06:46,  3.01s/it]
Epoch Progress:  87%|████████████████████▊   | 866/1000 [43:32<06:43,  3.01s/it]
Epoch Progress:  87%|████████████████████▊   | 867/1000 [43:35<06:40,  3.01s/it]
Epoch Progress:  87%|████████████████████▊   | 868/1000 [43:38<06:37,  3.01s/it]
Epoch Progress:  87%|████████████████████▊   | 869/1000 [43:41<06:34,  3.01s/it]
Epoch Progress:  87%|████████████████████▉   | 870/1000 [43:44<06:30,  3.01s/it]Dataset Size: 10%, Epoch: 871, Train Loss: 9.394398244222005, Val Loss: 10.188553872046533

Epoch Progress:  87%|████████████████████▉   | 871/1000 [43:47<06:27,  3.01s/it]
Epoch Progress:  87%|████████████████████▉   | 872/1000 [43:50<06:25,  3.01s/it]
Epoch Progress:  87%|████████████████████▉   | 873/1000 [43:53<06:21,  3.01s/it]
Epoch Progress:  87%|████████████████████▉   | 874/1000 [43:56<06:19,  3.01s/it]
Epoch Progress:  88%|█████████████████████   | 875/1000 [43:59<06:16,  3.01s/it]
Epoch Progress:  88%|█████████████████████   | 876/1000 [44:02<06:13,  3.01s/it]
Epoch Progress:  88%|█████████████████████   | 877/1000 [44:05<06:10,  3.01s/it]
Epoch Progress:  88%|█████████████████████   | 878/1000 [44:08<06:06,  3.01s/it]
Epoch Progress:  88%|█████████████████████   | 879/1000 [44:11<06:04,  3.01s/it]
Epoch Progress:  88%|█████████████████████   | 880/1000 [44:14<06:00,  3.01s/it]Dataset Size: 10%, Epoch: 881, Train Loss: 9.389519373575846, Val Loss: 10.187805324405819

Epoch Progress:  88%|█████████████████████▏  | 881/1000 [44:17<05:57,  3.01s/it]
Epoch Progress:  88%|█████████████████████▏  | 882/1000 [44:20<05:54,  3.00s/it]
Epoch Progress:  88%|█████████████████████▏  | 883/1000 [44:23<05:52,  3.01s/it]
Epoch Progress:  88%|█████████████████████▏  | 884/1000 [44:26<05:49,  3.02s/it]
Epoch Progress:  88%|█████████████████████▏  | 885/1000 [44:29<05:46,  3.01s/it]
Epoch Progress:  89%|█████████████████████▎  | 886/1000 [44:32<05:43,  3.01s/it]
Epoch Progress:  89%|█████████████████████▎  | 887/1000 [44:35<05:40,  3.01s/it]
Epoch Progress:  89%|█████████████████████▎  | 888/1000 [44:38<05:37,  3.01s/it]
Epoch Progress:  89%|█████████████████████▎  | 889/1000 [44:41<05:34,  3.01s/it]
Epoch Progress:  89%|█████████████████████▎  | 890/1000 [44:44<05:30,  3.01s/it]Dataset Size: 10%, Epoch: 891, Train Loss: 9.389763005574544, Val Loss: 10.189077563100048

Epoch Progress:  89%|█████████████████████▍  | 891/1000 [44:47<05:27,  3.01s/it]
Epoch Progress:  89%|█████████████████████▍  | 892/1000 [44:50<05:24,  3.01s/it]
Epoch Progress:  89%|█████████████████████▍  | 893/1000 [44:53<05:22,  3.01s/it]
Epoch Progress:  89%|█████████████████████▍  | 894/1000 [44:56<05:19,  3.01s/it]
Epoch Progress:  90%|█████████████████████▍  | 895/1000 [44:59<05:16,  3.01s/it]
Epoch Progress:  90%|█████████████████████▌  | 896/1000 [45:02<05:13,  3.01s/it]
Epoch Progress:  90%|█████████████████████▌  | 897/1000 [45:05<05:10,  3.01s/it]
Epoch Progress:  90%|█████████████████████▌  | 898/1000 [45:08<05:06,  3.01s/it]
Epoch Progress:  90%|█████████████████████▌  | 899/1000 [45:11<05:03,  3.01s/it]
Epoch Progress:  90%|█████████████████████▌  | 900/1000 [45:14<05:01,  3.01s/it]Dataset Size: 10%, Epoch: 901, Train Loss: 9.39577407836914, Val Loss: 10.189892174361589

Epoch Progress:  90%|█████████████████████▌  | 901/1000 [45:17<04:57,  3.01s/it]
Epoch Progress:  90%|█████████████████████▋  | 902/1000 [45:20<04:55,  3.01s/it]
Epoch Progress:  90%|█████████████████████▋  | 903/1000 [45:23<04:52,  3.01s/it]
Epoch Progress:  90%|█████████████████████▋  | 904/1000 [45:26<04:48,  3.01s/it]
Epoch Progress:  90%|█████████████████████▋  | 905/1000 [45:29<04:46,  3.02s/it]
Epoch Progress:  91%|█████████████████████▋  | 906/1000 [45:32<04:43,  3.02s/it]
Epoch Progress:  91%|█████████████████████▊  | 907/1000 [45:35<04:40,  3.01s/it]
Epoch Progress:  91%|█████████████████████▊  | 908/1000 [45:38<04:36,  3.01s/it]
Epoch Progress:  91%|█████████████████████▊  | 909/1000 [45:41<04:33,  3.01s/it]
Epoch Progress:  91%|█████████████████████▊  | 910/1000 [45:44<04:30,  3.01s/it]Dataset Size: 10%, Epoch: 911, Train Loss: 9.395861473083496, Val Loss: 10.188111602485954

Epoch Progress:  91%|█████████████████████▊  | 911/1000 [45:47<04:27,  3.01s/it]
Epoch Progress:  91%|█████████████████████▉  | 912/1000 [45:50<04:24,  3.00s/it]
Epoch Progress:  91%|█████████████████████▉  | 913/1000 [45:53<04:21,  3.00s/it]
Epoch Progress:  91%|█████████████████████▉  | 914/1000 [45:56<04:18,  3.00s/it]
Epoch Progress:  92%|█████████████████████▉  | 915/1000 [45:59<04:15,  3.01s/it]
Epoch Progress:  92%|█████████████████████▉  | 916/1000 [46:02<04:13,  3.02s/it]
Epoch Progress:  92%|██████████████████████  | 917/1000 [46:05<04:13,  3.05s/it]
Epoch Progress:  92%|██████████████████████  | 918/1000 [46:08<04:09,  3.04s/it]
Epoch Progress:  92%|██████████████████████  | 919/1000 [46:11<04:05,  3.03s/it]
Epoch Progress:  92%|██████████████████████  | 920/1000 [46:14<04:02,  3.03s/it]Dataset Size: 10%, Epoch: 921, Train Loss: 9.394017740885417, Val Loss: 10.187491516014198

Epoch Progress:  92%|██████████████████████  | 921/1000 [46:17<03:59,  3.03s/it]
Epoch Progress:  92%|██████████████████████▏ | 922/1000 [46:20<03:55,  3.02s/it]
Epoch Progress:  92%|██████████████████████▏ | 923/1000 [46:23<03:52,  3.02s/it]
Epoch Progress:  92%|██████████████████████▏ | 924/1000 [46:26<03:49,  3.02s/it]
Epoch Progress:  92%|██████████████████████▏ | 925/1000 [46:29<03:46,  3.01s/it]
Epoch Progress:  93%|██████████████████████▏ | 926/1000 [46:32<03:43,  3.02s/it]
Epoch Progress:  93%|██████████████████████▏ | 927/1000 [46:35<03:40,  3.02s/it]
Epoch Progress:  93%|██████████████████████▎ | 928/1000 [46:38<03:36,  3.01s/it]
Epoch Progress:  93%|██████████████████████▎ | 929/1000 [46:41<03:34,  3.01s/it]
Epoch Progress:  93%|██████████████████████▎ | 930/1000 [46:44<03:30,  3.01s/it]Dataset Size: 10%, Epoch: 931, Train Loss: 9.393095728556315, Val Loss: 10.190070944947081

Epoch Progress:  93%|██████████████████████▎ | 931/1000 [46:47<03:27,  3.01s/it]
Epoch Progress:  93%|██████████████████████▎ | 932/1000 [46:50<03:24,  3.01s/it]
Epoch Progress:  93%|██████████████████████▍ | 933/1000 [46:53<03:21,  3.01s/it]
Epoch Progress:  93%|██████████████████████▍ | 934/1000 [46:56<03:18,  3.01s/it]
Epoch Progress:  94%|██████████████████████▍ | 935/1000 [47:00<03:15,  3.01s/it]
Epoch Progress:  94%|██████████████████████▍ | 936/1000 [47:03<03:12,  3.01s/it]
Epoch Progress:  94%|██████████████████████▍ | 937/1000 [47:06<03:10,  3.02s/it]
Epoch Progress:  94%|██████████████████████▌ | 938/1000 [47:09<03:07,  3.02s/it]
Epoch Progress:  94%|██████████████████████▌ | 939/1000 [47:12<03:04,  3.02s/it]
Epoch Progress:  94%|██████████████████████▌ | 940/1000 [47:15<03:00,  3.01s/it]Dataset Size: 10%, Epoch: 941, Train Loss: 9.392686945597331, Val Loss: 10.18857495196454

Epoch Progress:  94%|██████████████████████▌ | 941/1000 [47:18<02:57,  3.01s/it]
Epoch Progress:  94%|██████████████████████▌ | 942/1000 [47:21<02:55,  3.02s/it]
Epoch Progress:  94%|██████████████████████▋ | 943/1000 [47:24<02:51,  3.02s/it]
Epoch Progress:  94%|██████████████████████▋ | 944/1000 [47:27<02:48,  3.01s/it]
Epoch Progress:  94%|██████████████████████▋ | 945/1000 [47:30<02:45,  3.01s/it]
Epoch Progress:  95%|██████████████████████▋ | 946/1000 [47:33<02:42,  3.01s/it]
Epoch Progress:  95%|██████████████████████▋ | 947/1000 [47:36<02:39,  3.02s/it]
Epoch Progress:  95%|██████████████████████▊ | 948/1000 [47:39<02:36,  3.01s/it]
Epoch Progress:  95%|██████████████████████▊ | 949/1000 [47:42<02:33,  3.02s/it]
Epoch Progress:  95%|██████████████████████▊ | 950/1000 [47:45<02:30,  3.01s/it]Dataset Size: 10%, Epoch: 951, Train Loss: 9.39506519317627, Val Loss: 10.185333697826831

Epoch Progress:  95%|██████████████████████▊ | 951/1000 [47:48<02:27,  3.01s/it]
Epoch Progress:  95%|██████████████████████▊ | 952/1000 [47:51<02:24,  3.01s/it]
Epoch Progress:  95%|██████████████████████▊ | 953/1000 [47:54<02:21,  3.01s/it]
Epoch Progress:  95%|██████████████████████▉ | 954/1000 [47:57<02:18,  3.01s/it]
Epoch Progress:  96%|██████████████████████▉ | 955/1000 [48:00<02:15,  3.01s/it]
Epoch Progress:  96%|██████████████████████▉ | 956/1000 [48:03<02:12,  3.01s/it]
Epoch Progress:  96%|██████████████████████▉ | 957/1000 [48:06<02:09,  3.01s/it]
Epoch Progress:  96%|██████████████████████▉ | 958/1000 [48:09<02:06,  3.02s/it]
Epoch Progress:  96%|███████████████████████ | 959/1000 [48:12<02:03,  3.02s/it]
Epoch Progress:  96%|███████████████████████ | 960/1000 [48:15<02:00,  3.01s/it]Dataset Size: 10%, Epoch: 961, Train Loss: 9.391535008748372, Val Loss: 10.18947680584796

Epoch Progress:  96%|███████████████████████ | 961/1000 [48:18<01:57,  3.01s/it]
Epoch Progress:  96%|███████████████████████ | 962/1000 [48:21<01:54,  3.02s/it]
Epoch Progress:  96%|███████████████████████ | 963/1000 [48:24<01:51,  3.01s/it]
Epoch Progress:  96%|███████████████████████▏| 964/1000 [48:27<01:48,  3.01s/it]
Epoch Progress:  96%|███████████████████████▏| 965/1000 [48:30<01:45,  3.01s/it]
Epoch Progress:  97%|███████████████████████▏| 966/1000 [48:33<01:42,  3.01s/it]
Epoch Progress:  97%|███████████████████████▏| 967/1000 [48:36<01:39,  3.01s/it]
Epoch Progress:  97%|███████████████████████▏| 968/1000 [48:39<01:36,  3.02s/it]
Epoch Progress:  97%|███████████████████████▎| 969/1000 [48:42<01:33,  3.02s/it]
Epoch Progress:  97%|███████████████████████▎| 970/1000 [48:45<01:30,  3.02s/it]Dataset Size: 10%, Epoch: 971, Train Loss: 9.390840746561686, Val Loss: 10.187597918820071

Epoch Progress:  97%|███████████████████████▎| 971/1000 [48:48<01:27,  3.01s/it]
Epoch Progress:  97%|███████████████████████▎| 972/1000 [48:51<01:24,  3.01s/it]
Epoch Progress:  97%|███████████████████████▎| 973/1000 [48:54<01:21,  3.01s/it]
Epoch Progress:  97%|███████████████████████▍| 974/1000 [48:57<01:18,  3.01s/it]
Epoch Progress:  98%|███████████████████████▍| 975/1000 [49:00<01:15,  3.01s/it]
Epoch Progress:  98%|███████████████████████▍| 976/1000 [49:03<01:12,  3.01s/it]
Epoch Progress:  98%|███████████████████████▍| 977/1000 [49:06<01:09,  3.01s/it]
Epoch Progress:  98%|███████████████████████▍| 978/1000 [49:09<01:06,  3.01s/it]
Epoch Progress:  98%|███████████████████████▍| 979/1000 [49:12<01:03,  3.02s/it]
Epoch Progress:  98%|███████████████████████▌| 980/1000 [49:15<01:00,  3.02s/it]Dataset Size: 10%, Epoch: 981, Train Loss: 9.388435478210448, Val Loss: 10.188839020667139

Epoch Progress:  98%|███████████████████████▌| 981/1000 [49:18<00:57,  3.02s/it]
Epoch Progress:  98%|███████████████████████▌| 982/1000 [49:21<00:54,  3.02s/it]
Epoch Progress:  98%|███████████████████████▌| 983/1000 [49:24<00:51,  3.02s/it]
Epoch Progress:  98%|███████████████████████▌| 984/1000 [49:27<00:48,  3.02s/it]
Epoch Progress:  98%|███████████████████████▋| 985/1000 [49:30<00:45,  3.01s/it]
Epoch Progress:  99%|███████████████████████▋| 986/1000 [49:33<00:42,  3.01s/it]
Epoch Progress:  99%|███████████████████████▋| 987/1000 [49:36<00:39,  3.01s/it]
Epoch Progress:  99%|███████████████████████▋| 988/1000 [49:39<00:36,  3.01s/it]
Epoch Progress:  99%|███████████████████████▋| 989/1000 [49:42<00:33,  3.02s/it]
Epoch Progress:  99%|███████████████████████▊| 990/1000 [49:45<00:30,  3.02s/it]Dataset Size: 10%, Epoch: 991, Train Loss: 9.391411552429199, Val Loss: 10.189635846521947

Epoch Progress:  99%|███████████████████████▊| 991/1000 [49:48<00:27,  3.01s/it]
Epoch Progress:  99%|███████████████████████▊| 992/1000 [49:51<00:24,  3.02s/it]
Epoch Progress:  99%|███████████████████████▊| 993/1000 [49:54<00:21,  3.03s/it]
Epoch Progress:  99%|███████████████████████▊| 994/1000 [49:57<00:18,  3.03s/it]
Epoch Progress: 100%|███████████████████████▉| 995/1000 [50:00<00:15,  3.02s/it]
Epoch Progress: 100%|███████████████████████▉| 996/1000 [50:03<00:12,  3.03s/it]
Epoch Progress: 100%|███████████████████████▉| 997/1000 [50:07<00:09,  3.03s/it]
Epoch Progress: 100%|███████████████████████▉| 998/1000 [50:10<00:06,  3.02s/it]
Epoch Progress: 100%|███████████████████████▉| 999/1000 [50:13<00:03,  3.03s/it]
Epoch Progress: 100%|███████████████████████| 1000/1000 [50:16<00:00,  3.02s/it]
Dataset Size: 10%, Val Perplexity: 26286.486328125
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
plt.title('Params=200k, Fraction=0.1, LR=0.001')
plt.legend()
plt.grid(True)
plt.savefig('200k_small_f_0.1_lr_0.001.png')
plt.show()
