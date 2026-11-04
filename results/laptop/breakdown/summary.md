# BMSSP time breakdown against Dijkstra

Medians over 5 sources (ms). Ratios are paired per source, then the median is taken.

| Graph | Dijkstra | BMSSP | BMSSP / Dijkstra | D operations | D ops / Dijkstra | Edge work (FindPivots + base + relax scan) | Other | Max BMSSP speedup if all edge work were free |
|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 25.6 | 139.8 | 5.47x | 70.1 | 2.69x | 59.0 | 9.8 | 1.75x |
| grid500 | 34.7 | 123.3 | 3.42x | 77.2 | 2.21x | 34.9 | 10.5 | 1.39x |
| rmat16 | 10.5 | 60.0 | 5.71x | 40.4 | 3.84x | 16.0 | 4.0 | 1.34x |
| rmat18 | 59.8 | 308.0 | 5.14x | 215.4 | 3.58x | 78.2 | 20.7 | 1.34x |
| rmat20 | 343.2 | 1570.4 | 5.12x | 1063.7 | 3.17x | 418.2 | 120.4 | 1.36x |

D operations by kind (replay, ms): initialize / insert pivots / pull / relax-loop erase+insert / batch prepend

- USA-road-d.NY: 0.4 / 3.9 / 21.4 / 42.7 / 2.4; in-run timers: pull 20.9, prepend 2.3
- grid500: 0.7 / 4.9 / 23.3 / 45.1 / 3.1; in-run timers: pull 23.7, prepend 3.0
- rmat16: 2.0 / 0.6 / 5.7 / 19.7 / 12.3; in-run timers: pull 6.6, prepend 13.7
- rmat18: 9.2 / 0.2 / 40.8 / 105.8 / 60.0; in-run timers: pull 41.7, prepend 61.3
- rmat20: 43.0 / 0.8 / 199.9 / 543.1 / 270.8; in-run timers: pull 180.4, prepend 289.2
