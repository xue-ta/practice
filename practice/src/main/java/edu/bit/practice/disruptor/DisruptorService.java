package edu.bit.practice.disruptor;

import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;
import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import com.lmax.disruptor.*;

import java.util.concurrent.Executors;

@Component
public class DisruptorService implements InitializingBean, DisposableBean {


    private Disruptor<StockInfo> disruptor;
    @Value("1024")
    private int RING_BUFFER_SIZE = 1024;
    @Value("10")
    private int EVENT_THREADS = 10;


    public void receive(StockInfo s) {
        disruptor.publishEvent((event,sequence,arg0)->{
            event.setStockName(arg0.getStockName());
            event.setId(arg0.getId());
            event.setStartPrice(arg0.getStartPrice());
            event.setEndPrice(arg0.getEndPrice());
        },s);
    }

    @Override
    public void destroy() throws Exception {
        disruptor.shutdown();
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        disruptor = new Disruptor<StockInfo>(
                () -> new StockInfo(),
                RING_BUFFER_SIZE,
                Executors.defaultThreadFactory(),
                ProducerType.MULTI,
                new BlockingWaitStrategy());
        disruptor.setDefaultExceptionHandler(new DisruptorExceptionHandler());
        disruptor.handleEventsWithWorkerPool(getMultiWorkerHandler(DisruptorEventHandler.class, EVENT_THREADS));
        disruptor.start();
    }

    private WorkHandler[] getMultiWorkerHandler(Class<? extends WorkHandler> clazz, int num) throws Exception {
        WorkHandler[] array = new WorkHandler[num];
        for (int i = 0; i < num; i++) {
            array[i] = clazz.newInstance();
        }
        return array;
    }


    private static class DisruptorEventHandler implements WorkHandler<StockInfo>, EventHandler<StockInfo> {

        public DisruptorEventHandler() {
        }


        @Override
        public void onEvent(StockInfo event, long sequence, boolean endOfBatch)
                throws Exception {
            this.onEvent(event);
        }

        @Override
        public void onEvent(StockInfo event) throws Exception {
            ;
        }
    }

    private static class DisruptorExceptionHandler implements ExceptionHandler {

        @Override
        public void handleEventException(Throwable throwable, long sequence, Object event) {
            ;
        }

        @Override
        public void handleOnStartException(Throwable throwable) {
            ;
        }

        @Override
        public void handleOnShutdownException(Throwable throwable) {
            ;
        }
    }
}
