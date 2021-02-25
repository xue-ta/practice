package edu.bit.practice.handler.stock;


import com.lmax.disruptor.dsl.Disruptor;
import edu.bit.practice.disruptor.DisruptorService;
import edu.bit.practice.handler.Handler;
import edu.bit.practice.repository.StockInfoRepository;
import edu.bit.practice.repository.dao.StockInfo;
import edu.bit.practice.repository.mapper.StockInfoMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class RepositoryHandler implements Handler<StockInfo> {
    @Autowired
    private StockInfoRepository stockInfoRepository;
    @Autowired
    private StockInfoMapper stockInfoMapper;
    @Autowired
    private DisruptorService disruptorService;

    @Override
    public void handleRequest(StockInfo s) {
        stockInfoMapper.save(s);
        //stockInfoRepository.save(s);
        disruptorService.receive(s);
    }
}
