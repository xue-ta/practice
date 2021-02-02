package edu.bit.practice.handler.stockhandler;


import edu.bit.practice.handler.Handler;
import edu.bit.practice.netty.NettyClient;
import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class NettyHandler implements Handler<StockInfo> {
    @Autowired
    private NettyClient nettyClient;
    @Override
    public void handleRequest(StockInfo stockInfo) {
        nettyClient.write(stockInfo);
    }
}
