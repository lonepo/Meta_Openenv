from circuitsynth import CircuitSynthEnv
import time

for task, target in [('squarewave-easy', 555), ('squarewave-medium', 1000), ('squarewave-hard', 2000)]:
    best_score = -1
    best_r = -1
    best_c = -1
    for r in range(10, 16):
        for c in range(15):
            env = CircuitSynthEnv(task_id=task, seed=42, mock_sim=False)
            env.reset()
            actions = [
                {'action_type': 'ADD_COMPONENT', 'component_type': 'VSOURCE', 'value_idx': 5, 'node_a': 'VCC', 'node_b': 'GND'},
                {'action_type': 'ADD_COMPONENT', 'component_type': 'RESISTOR', 'value_idx': 6, 'node_a': 'VCC', 'node_b': 'N1'}, # Rc = 784
                {'action_type': 'ADD_COMPONENT', 'component_type': 'RESISTOR', 'value_idx': 6, 'node_a': 'VCC', 'node_b': 'N2'}, 
                {'action_type': 'ADD_COMPONENT', 'component_type': 'RESISTOR', 'value_idx': r, 'node_a': 'VCC', 'node_b': 'N3'},
                {'action_type': 'ADD_COMPONENT', 'component_type': 'RESISTOR', 'value_idx': r, 'node_a': 'VCC', 'node_b': 'N4'},
                {'action_type': 'ADD_COMPONENT', 'component_type': 'CAPACITOR', 'value_idx': c, 'node_a': 'N1', 'node_b': 'N4'},
                {'action_type': 'ADD_COMPONENT', 'component_type': 'CAPACITOR', 'value_idx': c, 'node_a': 'N2', 'node_b': 'N3'},
                {'action_type': 'ADD_COMPONENT', 'component_type': 'NPN_BJT', 'value_idx': 0, 'node_a': 'N1', 'node_b': 'N3', 'node_c': 'GND'},
                {'action_type': 'ADD_COMPONENT', 'component_type': 'NPN_BJT', 'value_idx': 0, 'node_a': 'N2', 'node_b': 'N4', 'node_c': 'GND'},
                {'action_type': 'FINALIZE'}
            ]
            try:
                for act in actions:
                    obs, reward, term, trunc, info = env.step_dict(act)
                if reward > best_score:
                    best_score = reward
                    best_r = r
                    best_c = c
            except:
                pass
    print(f'{task}: Best r_idx={best_r}, c_idx={best_c}, score={best_score:.4f}')

