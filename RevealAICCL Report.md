# RevealAICCL Report:

不确定的AllToAllv流量在Router运行后逐步产生，而传统scheduler通常假设完整traffic已知。

针对这种情况我们用下列办法

### 1.Revealed-Only的调度

##### （1）我们只根据即时到达的，已经revealed的流量做出schedule。

```AICCL
完整traffic matrix -> scheduler -> schedule
```

真实 MoE 状态：

```
Router chunk 0 → 知道一部分 token 去向
Router chunk 1 → 再知道一部分
...
Router final   → 才知道完整 traffic
```

比如截至时间$t_n$我们知道Router chunk 0~n的token去向



举例，我们现在有token0~7，将八个token分成2组，chunk 0和chunk 1

```
chunk 0:
t0 t1 t2 t3

chunk 1:
t4 t5 t6 t7
```

（关于chunk size：在提前训练过程中确定，不是越小越好）

##### （2）计算top-k

假设

```
t0 → Expert 2 → GPU1
t1 → Expert 0 → GPU0
t2 → Expert 3 → GPU1
t3 → Expert 1 → GPU0

t4 → Expert 2 → GPU1
t5 → Expert 2 → GPU1
t6 → Expert 0 → GPU0
t7 → Expert 3 → GPU1
```

GPU计算Router(chunk 0) ——top-k

则暂时我们只知道上面四个token的去向

##### （3）状态保证

每个chunk后有一个CUDA Event用来记录是否已经完成，完成后可进入scheduler



### 2.AICCL Scheduler

##### （1）维护remote demand的状态

比如chunk 0进来后新增 GPU0 -> GPU1 = 2 tokens

而之前还有x个tokens的没做完，则有

```
previous revealed = x
new delta = 2
```

##### （2）用提早规划的路径避免每次都要做BFS

比如拓扑固定，GPU a 到 GPU u的路径不会变动，每次只需要直接使用记录的路径即可

（变化的情况，考虑隔一段时间再修改一下BFS？）（暂时没有这一条）

物理GPU的拓扑正常情况不变

##### （3）提前准备通信模板 供revealed demand快速填入

##### （4）再次检查操作是否合法

##### （5）得到committed action

​	即这个chunk的schedule已做好



### 3.Communication Backend

比如我们现在有

```
Action:
GPU0 → GPU1
[t0,t2]
```

##### （1）Def

sendcounts = [ $a_0,a_1,...,a_x$]

sendcounts[$i$] = 给序号为$i$的GPU发多少数据

...这里先省略了

##### （2）调用接口

```
torch.distributed.all_to_all_single(
    ...,
    input_split_sizes=sendcounts,
    output_split_sizes=recvcounts,
    async_op=True
)
```



### 4.现在我们算完了chunk 0

##### （1）现在第二批：

```
t4 → GPU1
t5 → GPU1
t6 → GPU0
t7 → GPU1
```

```
Schedule =

Action 0:
GPU0 → GPU1 : [t0,t2]

Action 1:
GPU0 → GPU1 : [t4,t5,t7]

...
```

##### （2）最终长啥样

```
Router chunk0 ███
                 ↓ reveal
                 schedule0
                 comm0 ──────

Router chunk1      ███
                      ↓ reveal
                      schedule1
                      comm1 ──────

Router chunk2         ███
                         ↓
                         schedule2
                         comm2
```



### 5.目前的一个问题

MSCCL要求提前知道整个collective长成什么样子

在最新的MSCCL++中，官方 MSCCL++ 提供 GPU-side `Channel`，在 kernel 里可以直接调用类似：

```
put(...)
get(...)
signal(...)
wait(...)
flush(...)
```

和我目前的思路比较符合，若要和组内的PCCL对接，需要我们向MSCCL++靠一靠。
