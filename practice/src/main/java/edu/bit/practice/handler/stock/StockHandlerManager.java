package edu.bit.practice.handler.stock;

import edu.bit.practice.handler.Handler;
import edu.bit.practice.handler.HandlerChain;
import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;


@Component
public class StockHandlerManager implements HandlerChain {


    @Autowired
    private List<Handler<StockInfo>> l;

    @Override
    public StockHandlerManager addLast(Handler handler){
        l.add(handler);
        return this;
    }

    public void handle(StockInfo t){
        l.forEach(s->s.handleRequest(t));
    }
}
