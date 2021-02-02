package edu.bit.practice.handler.stock;


import edu.bit.practice.handler.Handler;
import edu.bit.practice.repository.StockInfoRepository;
import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class RepositoryHandler implements Handler<StockInfo> {
    @Autowired
    private StockInfoRepository stockInfoRepository;

    @Override
    public void handleRequest(StockInfo s) {
        stockInfoRepository.save(s);
    }
}
