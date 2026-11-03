| Model | ONFI input copies | Command ns | Cyclic MoE+shared ms | Whole MoE+shared ms | MoE gain | Complete FFN gain |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 1 | 250 | 52.357222 | 50.603891 | 3.3488% | 3.2344% |
| deepseek | 1 | 5000 | 55.196110 | 54.122153 | 1.9457% | 1.8779% |
| deepseek | 4 | 250 | 53.458011 | 51.655093 | 3.3726% | 3.2575% |
| deepseek | 4 | 5000 | 56.214576 | 55.127055 | 1.9346% | 1.8677% |
| granite4tiny | 1 | 250 | 24.510604 | 23.863417 | 2.6404% | 2.6404% |
| granite4tiny | 1 | 5000 | 28.796653 | 28.796653 | 0.0000% | 0.0000% |
| granite4tiny | 4 | 250 | 25.136775 | 24.991046 | 0.5797% | 0.5797% |
| granite4tiny | 4 | 5000 | 29.364461 | 29.364461 | 0.0000% | 0.0000% |

Uncalibrated service model; complete FFN adds DeepSeek raw0 dense only. Attention/router/SSM and host row-scale/gather excluded. No heldout fitting.
