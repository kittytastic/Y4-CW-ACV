from typing import Any, Dict
import json
import os

class StageState():
    def __init__(self, full_state:'State', stage:str) -> None:
        self.full_state = full_state
        self.stage = stage

    def update_fast(self, key:str, value:Any):
        self.full_state.update_fast(self.stage, key, value)

    def save(self):
        self.full_state.save()

    def __getitem__(self, key:str)->Any:
        return self.full_state.state[self.stage][key]
    
    def __setitem__(self, key:str, value: Any):
        self.full_state.update_and_save(self.stage, key, value)

    def __str__(self) -> str:
        return self.full_state._state_stage_str(self.stage, 0)
        


class StateTemplate():
    def __init__(self) -> None:
        self.default_state:Dict[str, Dict[str, Any]] = {}

    def register_stage(self, stage:str, stage_default_state:  Dict[str, Any]):
        assert stage not in self.default_state.keys()
        self.default_state[stage] = stage_default_state

class State:
    def __init__(self, state_template:StateTemplate, store_path: str) -> None:
        self.state:Dict[str, Dict[str, Any]] = {}
        self.default_state = state_template.default_state
        self.store_path = store_path
        self.restore(self.store_path)

    def update_and_save(self, stage:str, key:str, value:Any):
        self.update_fast(stage, key, value)
        self.save()

    def update_fast(self, stage:str, key:str, value:Any):
        self.state[stage][key] = value 

    def save(self):
        with open(self.store_path, "w+") as f :
            s = json.dumps(self.state)
            f.write(s)

    def restore(self, path:str):
        if not os.path.exists(path):
            self.restore_to_default()
            self.save()
        else:
            with open(self.store_path, "r") as f :
                s = f.read()
                self.state = json.loads(s)
                self._assert_state_matches_template()

    def _assert_state_matches_template(self):
        for stage, stage_state in self.default_state.items():
            assert stage in self.state, f"Expected stage: {stage} in state, but it wasn't found"
            for stage_key in stage_state.keys():
                assert stage_key in self.state[stage], f"Expected {stage}->{stage_key} to be in state, but it wasn't found"

    def restore_to_default(self):
        self.state = {}
        for stage, stage_state in self.default_state.items():
            self.state[stage] = dict(stage_state)
            

    def __getitem__(self, key:str):
        assert key in self.state.keys(), f"{key} in not one of the know states: {self.state.keys()}"
        return StageState(self, key)
    
    def _state_stage_str(self, stage:str, indent:int):
        indent_s= '\t'*indent
        return "\n".join([f"{indent_s}{k}: {v}" for k, v in self.state[stage].items()])

    def __str__(self):
        out_s = ""
        for stage in self.state.keys():
            out_s += f"{stage}\n"
            out_s += self._state_stage_str(stage, 1)
            out_s += "\n"
        
        return out_s