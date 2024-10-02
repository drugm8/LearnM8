from learners import abc_learner as learner

class chemprop_gpu_learner(learner):

    def __init__(self, query_function, dataset_x, dataset_y, batch_size=32):

        self.query_function = query_function
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        self.mpnn = do_chempop_gpu(smiles= self.dataset_x, ys=self.dataset_y ) #!ys can be multiple i think
        self.name = "chemprop_gpu_high epoch"

    def figure_out_system_and_set_parameters_accordingly(self):
        pass

    def set_int_batch_size(self, batch_size):
        self.batch_size = batch_size

    def teach(self, addition_of_dataset_x, addition_of_dataset_y):
        print("teaching...")
        self.dataset_y=np.append(addition_of_dataset_y, self.dataset_y)
        self.dataset_x=np.append(addition_of_dataset_x, self.dataset_x)
        self.mpnn = do_chempop_gpu(smiles= self.dataset_x, ys=self.dataset_y )
        print("done teaching...")

    
    def query(self, smids_x_input, path):
        #uses the intrinisc query function to run the inference first and then query the dataset

        estimation = self.estimate(smids_x_input.loc[:,"SMILES"])

        queried = self.query_function(smids_x_input, estimation, batch_size=self.batch_size)
        return queried


    def estimate(self, x_input):
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in x_input]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dset, num_workers=8, batch_size=256, shuffle=False) #!!!! ohmygod shuffle
        with torch.inference_mode():
            trainerr = pl.Trainer(
                enable_checkpointing=True,
                enable_progress_bar=False,
                accelerator="gpu",
                devices=1
            )
        predictions = trainerr.predict(self.mpnn, test_loader)
        ret = np.concatenate(predictions, axis=0)


        return ret
    
