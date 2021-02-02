package edu.bit.practice.handler;

import com.alibaba.fastjson.support.odps.udf.CodecCheck;
import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;


@Component
public class HandlerManager {


    @Autowired
    private List<Handler<StockInfo>> l;

    public HandlerManager addLast(Handler handler){
        l.add(handler);
        return this;
    }

    public void handle(StockInfo t){
        l.forEach(s->s.handleRequest(t));
    }
}
