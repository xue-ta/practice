package edu.bit.practice.repository.mapper;

import edu.bit.practice.repository.dao.StockInfo;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface StockInfoMapper {
    void save(StockInfo stockInfo);
}
