package edu.bit.practice.repository.dao;


import lombok.Data;

import javax.persistence.*;

@Entity
@Table(name ="stock_info")
@Data
public class StockInfo {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Integer id;

    @Column(name = "stock_name")
    private String stockName;

    @Column(name="start_price")
    private String startPrice;

    @Column(name="end_price")
    private String endPrice;
}
