package edu.bit.practice.repository;


import edu.bit.practice.repository.dao.StockInfo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StockInfoRepository extends JpaRepository<StockInfo, Integer> {
}
