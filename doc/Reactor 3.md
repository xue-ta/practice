

### 响应式编程 

**响应式编程=数据流+变化传递+声明式**

具备“异步非阻塞”特性和“流量控制”能力的数据流，称之为响应式流（Reactive Stream）

异步非阻塞 多线程+回调函数

流量控制 阻塞队列

### Reactor

**Reactor 是响应式编程范式的实现**

-  **编排性 Composability ** 以及 **可读性Readability**
- 使用丰富的 **操作符 Operator** 来处理形如 **流 Stream ** 数据
- **订阅 subscribe ** 之前什么都不会发生
- **背压 backpressure ** *消费者能够反向告知生产者生产内容的速度的能力*
- **高层次** 的抽象，从而达到 *并发无关* 的效果

#### Mono

一个 `Mono` 对象代表一个包含 零/一个（0..1）元素的结果，使用上可以类比 `Optional`

#### flux

一个 `Flux` 对象代表一个包含 0..N 个元素的响应式序列，使用上可以类比 `Stream`

#### subscribeOn

**影响源头的线程执行环境**

#### publishOn

 **改变后续的操作符的执行所在线程**
 
 ### webflux
 ### SpringCloud Gateway
