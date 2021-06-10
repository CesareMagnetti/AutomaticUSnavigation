from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                # NOTE: download the resized (and compressed) val set from
                # http://efrosgans.eecs.berkeley.edu/CUT/datasets/cityscapes_val_for_CUT.tar
                checkpoints_dir="/vol/bitbucket/cm1320/general/CUT/checkpoints/",
                results_dir="/vol/bitbucket/cm1320/general/CUT/results/",
                dataroot="/vol/bitbucket/cm1320/general/datasets/cityscapes_val/",
                # gpu_id="-1",
                # gpu_ids="-1",
                direction="BtoA",
                phase="val",
                name="cityscapes_cut_pretrained",
                CUT_mode="CUT",
            ),

            # Command 1
            Options(
                checkpoints_dir="/vol/bitbucket/cm1320/general/CUT/checkpoints/",
                results_dir="/vol/bitbucket/cm1320/general/CUT/results/",
                dataroot="/vol/bitbucket/cm1320/general/datasets/cityscapes_val/",
                # gpu_id="-1",
                # gpu_ids="-1",
                direction="BtoA",
                phase="val",
                name="cityscapes_fastcut_pretrained",
                CUT_mode="FastCUT",
            ),

            # Command 2
            Options(
                checkpoints_dir="/vol/bitbucket/cm1320/general/CUT/checkpoints/",
                results_dir="/vol/bitbucket/cm1320/general/CUT/results/",
                dataroot="/vol/bitbucket/cm1320/general/datasets/horse2zebra/",
                # gpu_id="-1",
                # gpu_ids="-1",
                name="horse2zebra_cut_pretrained",
                CUT_mode="CUT"
            ),

            # Command 3
            Options(
                checkpoints_dir="/vol/bitbucket/cm1320/general/CUT/checkpoints/",
                results_dir="/vol/bitbucket/cm1320/general/CUT/results/",
                dataroot="/vol/bitbucket/cm1320/general/datasets/horse2zebra/",
                # gpu_id="-1",
                # gpu_ids="-1",
                name="horse2zebra_fastcut_pretrained",
                CUT_mode="FastCUT",
            ),

            # Command 4
            Options(
                checkpoints_dir="/vol/bitbucket/cm1320/general/CUT/checkpoints/",
                results_dir="/vol/bitbucket/cm1320/general/CUT/results/",
                dataroot="/vol/bitbucket/cm1320/general/datasets/cat2dog/",
                # gpu_id="-1",
                # gpu_ids="-1",
                name="cat2dog_cut_pretrained",
                CUT_mode="CUT"
            ),

            # Command 5
            Options(
                checkpoints_dir="/vol/bitbucket/cm1320/general/CUT/checkpoints/",
                results_dir="/vol/bitbucket/cm1320/general/CUT/results/",
                dataroot="/vol/bitbucket/cm1320/general/datasets/cat2dog/",
                # gpu_id="-1",
                # gpu_ids="-1",
                name="cat2dog_fastcut_pretrained",
                CUT_mode="FastCUT",
            ),

            
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        return ["python test.py " + str(opt.set(num_test=500)) for opt in self.common_options()]
