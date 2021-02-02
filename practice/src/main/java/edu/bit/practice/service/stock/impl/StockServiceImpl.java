package edu.bit.practice.service.stock.impl;

import com.alibaba.fastjson.JSONObject;
import edu.bit.practice.handler.Handler;
import edu.bit.practice.handler.HandlerManager;
import edu.bit.practice.handler.stockhandler.NettyHandler;
import edu.bit.practice.handler.stockhandler.RepositoryHandler;
import edu.bit.practice.netty.NettyClient;
import edu.bit.practice.repository.StockInfoRepository;
import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;


@Service
public class StockServiceImpl {

    private static final String STOCK_INFO="http://hq.sinajs.cn/list={list}";
    private static final String PING_AN_STOCK_CODE="sh601318";


    @Autowired
    private RestTemplate restTemplate;
    @Autowired
    private HandlerManager handlerManager;



    @Scheduled(cron="0/5 * * * * ?")
    public void getStockInfo(){

        Map param=new HashMap();
        param.put("list",PING_AN_STOCK_CODE);
        ResponseEntity<String> res=restTemplate.getForEntity(STOCK_INFO,String.class,param);
        String[] info=res.getBody().substring(21,46).split(",");

        StockInfo stockInfo=toStockInfo(info);

        handlerManager.handle(stockInfo);
    }

    private StockInfo toStockInfo(String[] info){
        StockInfo stockInfo =new StockInfo();
        stockInfo.setStockName(info[0]);
        stockInfo.setStartPrice(info[2]);
        stockInfo.setEndPrice(info[3]);
        return stockInfo;
    }
}
