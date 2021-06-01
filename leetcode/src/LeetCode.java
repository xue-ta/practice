import java.util.*;

public class LeetCode {

    public LeetCode() {
    }

    //1086. 前五科的均分
    public int[][] highFive(int[][] items) {

        Map<Integer, PriorityQueue<Integer>> m=new HashMap<>();
        for (int[] item:items){
            PriorityQueue p=m.getOrDefault(item[0],new PriorityQueue<>(100, new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return o2-o1;
                }
            }));
            p.add(item[1]);
            m.putIfAbsent(item[0],p);
        }
        int[][] result=new int[m.size()][2];

        int cnt=0;
        for(Map.Entry<Integer,PriorityQueue<Integer>> e:m.entrySet()){
            int sum=0;
            for(int i=0;i<5;i++){
                sum=sum+e.getValue().poll();
            }
            result[cnt][0]=e.getKey();
            result[cnt][1]=(sum/5);
            cnt++;
        }

        return result;
    }


    public static void main(String[] args) {

        int[][] items={{1,91},{1,92},{2,93},{2,97},{1,60},{2,77},{1,65},{1,87},{1,100},{2,100},{2,76}};

        LeetCode lt=new LeetCode();
        lt.highFive(items);

    }
}
