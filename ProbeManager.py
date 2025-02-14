import torch

class ProbeManager:
    def __init__(self, N_inputs = None, create_list_of_lists = False) -> None:
        # Flag to indicate if a list of lists should be created
        self._list_of_lists = create_list_of_lists

        # Create a list of empty entries
        if N_inputs != None:
            self._probe_in = list()
            self._probe_out = list()

            if create_list_of_lists:
                for i in range(N_inputs):
                    self._probe_in.append(list())
                    self._probe_out.append(list()) 
            else:
                for i in range(N_inputs):
                    self._probe_in.append(None)
                    self._probe_out.append(None) 
        else:
            self._probe_in = None
            self._probe_out = None

    def add_probe(self, input_id, input_val, output_val) -> None:
        # Save the input probe
        self._probe_in[input_id] = input_val

        # Save the output probe (Note: we use input_id as index)
        self._probe_out[input_id] = output_val

    def collect_probe(self, input_id, input_val, output_val) -> None:
        if self._list_of_lists == False:
            RuntimeError("ProbeManager: Cannot complete this operation since list_of_lists flag is False")

        in_list = self._probe_in[input_id]
        out_list = self._probe_out[input_id]

        in_list.append(input_val)
        out_list.append(output_val)

    def save(self, epoch_or_batch, probe_dir, probe_name, save_epoch=True) -> None:
        # Create the probe directory for the current epoch
        if save_epoch:
            epoch_batch_probe_dir = probe_dir / f"epoch_{epoch_or_batch:02d}"
        else:
            epoch_batch_probe_dir = probe_dir / f"batch_{epoch_or_batch:02d}"

        if epoch_batch_probe_dir.exists() == False:
            epoch_batch_probe_dir.mkdir(parents=True)

        probe_fname = epoch_batch_probe_dir / f"{probe_name}.pt"
        probe_dict = {'input' : self._probe_in, 'output' : self._probe_out, 'list_of_lists' : self._list_of_lists}
        torch.save(probe_dict, probe_fname)


    def clear(self) -> None:
        # Clear the list entries
        N_inputs = len(self._probe_in)

        if self._list_of_lists:
            for i in range(N_inputs):
                in_list = self._probe_in[i]
                in_list.clear()

                out_list = self._probe_out[i]
                out_list.clear()
        else:
            self._probe_in.clear()
            self._probe_out.clear()

            for i in range(N_inputs):
                self._probe_in.append(None)
                self._probe_out.append(None) 


    def load(self, epoch_or_batch, probe_dir, probe_name, load_epoch=True) -> None:
        if load_epoch:
            epoch_batch_probe_dir = probe_dir / f"epoch_{epoch_or_batch:02d}"
        else:
            epoch_batch_probe_dir = probe_dir / f"batch_{epoch_or_batch:02d}"

        probe_fname = epoch_batch_probe_dir / f"{probe_name}.pt"
        if probe_fname.exists() == True:
            probe_dict = torch.load(probe_fname)
            self._probe_in = probe_dict["input"]
            self._probe_out = probe_dict["output"]
            self._list_of_lists = probe_dict["list_of_lists"]
        else:
            raise ValueError(f"Probe file {probe_fname} not found")

