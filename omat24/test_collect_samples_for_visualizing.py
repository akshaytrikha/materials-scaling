# External
from pathlib import Path
import unittest
import torch
import numpy as np


# Internal
from data import OMat24Dataset, get_dataloaders
from data_utils import download_dataset
from models.fcn import FCNModel
from models.schnet import SchNet
from train_utils import collect_samples_for_visualizing


class TestCollectTrainValSamples(unittest.TestCase):
    def setup(self, graph: bool):
        SEED = 1024
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cpu")

        # Load dataset
        split_name = "val"
        dataset_name = "rattled-300-subsampled"

        dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, split_name)
        dataset = OMat24Dataset(dataset_path=dataset_path, augment=False, graph=graph)

        self.train_loader, self.val_loader = get_dataloaders(
            dataset,
            train_data_fraction=0.001,
            batch_size=2,
            seed=1024,
            batch_padded=False,
            val_data_fraction=0.001,
            graph=graph,
        )
        if graph:
            self.model = SchNet(
                hidden_channels=64,
                num_filters=64,
                num_interactions=3,
                num_gaussians=50,
            )
        else:
            # 1060 params
            self.model = FCNModel(
                vocab_size=119,
                embedding_dim=6,
                hidden_dim=6,
                depth=4,
                use_factorized=False,
            )
        self.num_visualization_samples = 3

    def test_structure_and_content_non_graph(self):
        """Test sample structure and content for non-graph mode."""
        self.setup(graph=False)

        samples = collect_samples_for_visualizing(
            self.model,
            False,  # non-graph mode
            self.train_loader,
            self.val_loader,
            self.device,
            self.num_visualization_samples,
        )
        # Expect 3 samples per split
        self.assertEqual(len(samples["train"]), 3)
        self.assertEqual(len(samples["val"]), 3)

        # Only check that the first train sample has the expected structure and content
        sample = samples["train"][0]

        for key in [
            "idx",
            "symbols",
            "atomic_numbers",
            "positions",
            "true",
            "pred",
        ]:
            self.assertIn(key, sample)

        self.assertEqual(
            sample["atomic_numbers"], [38, 38, 39, 39, 39, 39, 48, 48, 48, 48]
        )
        np.testing.assert_allclose(
            sample["positions"],
            [
                [0.12072731554508209, 8.525409698486328, 0.4263860881328583],
                [5.0489182472229, 4.623912811279297, 3.6889312267303467],
                [1.8813209533691406, 5.755345821380615, 1.7509342432022095],
                [6.130558490753174, 7.964021682739258, 1.371782660484314],
                [8.524673461914062, 2.6005146503448486, 1.2754912376403809],
                [2.8396027088165283, 1.0804824829101562, 2.0712502002716064],
                [2.801896810531616, 8.39545726776123, 3.805094003677368],
                [8.139731407165527, 6.2843546867370605, 0.415322870016098],
                [5.908754348754883, 1.6081221103668213, 3.1987249851226807],
                [2.1846492290496826, 3.516913652420044, 3.6767876148223877],
            ],
            rtol=1e-5,
            atol=1e-6,
        )

        true_vals = sample["true"]
        # Expect forces of shape (3, 2) coming from the transposed dataset forces:
        np.testing.assert_allclose(
            true_vals["forces"],
            [
                [
                    -0.9071914553642273,
                    1.8023358583450317,
                    -0.035239748656749725,
                ],
                [0.6905606985092163, 0.4602694809436798, -0.01464011985808611],
                [
                    -0.26647257804870605,
                    0.4796809256076813,
                    0.032321520149707794,
                ],
                [-1.368228793144226, 1.4275697469711304, 0.8258275389671326],
                [-0.0644230917096138, 0.45888352394104004, 0.132628932595253],
                [0.358563095331192, 1.876037359237671, -1.7133018970489502],
                [2.1366045475006104, -2.4676804542541504, 0.7717555165290833],
                [-0.04102186858654022, -2.773374557495117, -0.7009332180023193],
                [0.5296327471733093, -0.7196695804595947, 0.10698127746582031],
                [-1.0680232048034668, -0.5440521836280823, 0.5946001410484314],
            ],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertAlmostEqual(true_vals["energy"], -28.98720359802246, places=4)
        np.testing.assert_allclose(
            true_vals["stress"],
            [
                -0.0013863879721611738,
                -0.015082982368767262,
                0.01023661345243454,
                -0.0004944438696838915,
                0.006699948571622372,
                0.0008979742997325957,
            ],
            rtol=1e-5,
            atol=1e-6,
        )

        pred_vals = sample["pred"]
        np.testing.assert_allclose(
            pred_vals["forces"],
            [
                [-0.6780799627304077, -1.5160281658172607, 0.4360101819038391],
                [-0.75079345703125, 0.8309316039085388, -0.5847321152687073],
                [
                    -0.33938905596733093,
                    0.3369714617729187,
                    -0.18498724699020386,
                ],
                [
                    0.0011460334062576294,
                    1.3206708431243896,
                    -0.7868089079856873,
                ],
                [-0.5337921380996704, 2.281526803970337, -1.2020928859710693],
                [-0.8850125074386597, 1.1606290340423584, -0.5226950645446777],
                [-0.13733404874801636, 1.0548714399337769, -0.6875092387199402],
                [-0.1676117181777954, 1.4901158809661865, -0.5572907328605652],
                [-0.8274887800216675, 1.7359392642974854, -0.7181147933006287],
                [-0.7254419326782227, 0.8863802552223206, -0.33576011657714844],
            ],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertAlmostEqual(pred_vals["energy"], 3.7296910285949707, places=4)
        np.testing.assert_allclose(
            pred_vals["stress"],
            [
                -12.846657752990723,
                10.06513500213623,
                -11.453512191772461,
                11.033711433410645,
                -16.870542526245117,
                -21.56986427307129,
            ],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_structure_and_content_graph(self):
        """Test sample structure and content for graph mode."""
        self.setup(graph=True)

        samples = collect_samples_for_visualizing(
            self.model,
            True,  # graph mode
            self.train_loader,
            self.val_loader,
            self.device,
            self.num_visualization_samples,
        )
        # Expect 3 samples per split
        self.assertEqual(len(samples["train"]), 3)
        self.assertEqual(len(samples["val"]), 3)

        # Only check that the first validation sample has the expected structure and content
        sample = samples["val"][0]

        # atomic_numbers should now include all entries
        self.assertEqual(
            sample["atomic_numbers"],
            [12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 26, 26, 26, 26],
        )
        np.testing.assert_allclose(
            sample["positions"],
            [
                [6.501579284667969, 2.482605457305908, 0.17546893656253815],
                [10.52541446685791, 6.430670261383057, 1.3528333902359009],
                [6.604191780090332, 6.063041687011719, 2.6255598068237305],
                [10.527249336242676, 10.202180862426758, 1.6697824001312256],
                [10.951179504394531, 5.614882946014404, 3.4311044216156006],
                [5.817686557769775, 9.533682823181152, 3.50799298286438],
                [10.64077091217041, 10.02700424194336, 4.791032791137695],
                [13.884686470031738, 14.976096153259277, 5.004591941833496],
                [12.600369453430176, 12.747586250305176, 5.275933265686035],
                [3.9640486240386963, 7.301289081573486, 0.9946029186248779],
                [8.873032569885254, 8.206246376037598, 3.5990960597991943],
                [8.214850425720215, 12.366679191589355, 4.054942607879639],
                [7.968503475189209, 8.288232803344727, 5.210590839385986],
                [8.45325756072998, 5.113421440124512, 1.7743555307388306],
                [12.942307472229004, 12.719608306884766, 3.583418130874634],
                [12.5682954788208, 8.923619270324707, 4.4600324630737305],
            ],
            rtol=1e-5,
            atol=1e-6,
        )

        true_vals = sample["true"]
        np.testing.assert_allclose(
            true_vals["forces"],
            [
                [0.39692559838294983, 0.8447110056877136, 2.145308494567871],
                [-1.6965211629867554, 0.9914994835853577, -3.532607316970825],
                [-1.9057790040969849, 0.9068858027458191, 0.5525928735733032],
                [-0.6740679740905762, -0.12921838462352753, 1.1806145906448364],
                [-1.152750849723816, -0.02642575092613697, 2.9620842933654785],
                [2.152494192123413, -0.5634263157844543, -2.8250515460968018],
                [-1.4934406280517578, 1.520157814025879, -0.5385276675224304],
                [0.9204066395759583, 0.7224070429801941, 0.0994606614112854],
                [-2.246216058731079, 0.15408992767333984, 6.116621971130371],
                [1.633126139640808, 1.0323141813278198, -0.5072864890098572],
                [0.5769804120063782, -0.7168604135513306, -1.3577898740768433],
                [-0.020680749788880348, 0.36608996987342834, -0.6827691793441772],
                [-0.7543104290962219, 0.33542779088020325, 1.2344539165496826],
                [1.401350498199463, -1.9720216989517212, 0.009758800268173218],
                [3.0335588455200195, -1.259576678276062, -5.405921459197998],
                [-0.17107553780078888, -2.2060537338256836, 0.54905766248703],
            ],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertAlmostEqual(true_vals["energy"], -57.276607513427734, places=6)
        np.testing.assert_allclose(
            true_vals["stress"],
            [
                0.0009602017817087471,
                0.04748936742544174,
                -0.0102050406858325,
                -0.004416473209857941,
                0.02914390340447426,
                0.00938539020717144,
            ],
            rtol=1e-5,
            atol=1e-6,
        )

        pred_vals = sample["pred"]
        np.testing.assert_allclose(
            pred_vals["forces"],
            [
                [
                    -3.38680183631368e-05,
                    -2.6609601263771765e-05,
                    -1.4655783161288127e-05,
                ],
                [
                    0.00015130020619835705,
                    -0.00027923379093408585,
                    2.677248266991228e-05,
                ],
                [6.18219783063978e-05, -0.00015120544412638992, -1.39865733217448e-06],
                [7.066409307299182e-05, -3.5100034438073635e-05, 4.598424129653722e-05],
                [
                    0.0003826403117273003,
                    -0.0003470162919256836,
                    -0.00040546804666519165,
                ],
                [
                    -0.0002622628817334771,
                    -5.1911429181927815e-05,
                    7.800573075655848e-05,
                ],
                [
                    -0.00023987976601347327,
                    6.416856194846332e-05,
                    -0.0003507291548885405,
                ],
                [8.029423770494759e-05, 0.00025154242757707834, 0.00027568743098527193],
                [0.00043513166019693017, 0.0003121250483673066, -0.0010156125063076615],
                [-0.0001941709779202938, 4.952471863362007e-05, -0.000138252682518214],
                [0.0013612289912998676, 0.0007370631210505962, -0.0013885961379855871],
                [
                    7.677853864151984e-05,
                    -0.00010419067984912544,
                    -1.1125813216494862e-05,
                ],
                [
                    -0.0017924420535564423,
                    -0.00016606520512141287,
                    0.0019055006559938192,
                ],
                [0.0001959451474249363, 0.0004558713117148727, 0.00037357595283538103],
                [
                    -0.0003535982104949653,
                    -0.00042670490802265704,
                    0.0007934362511150539,
                ],
                [
                    6.041663436917588e-05,
                    -0.00028225782443769276,
                    -0.00017312410636804998,
                ],
            ],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertAlmostEqual(pred_vals["energy"], -7.751392331556417e-06, places=6)
        np.testing.assert_allclose(
            pred_vals["stress"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rtol=1e-5,
            atol=1e-6,
        )
