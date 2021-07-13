import java.util.*;

public class LeetCode {


public static class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode() {}
     TreeNode(int val) { this.val = val; }
      TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
 }


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


    public String reverseWords(String s) {
        StringBuilder sb=new StringBuilder();
        LinkedList<Character> stack=new LinkedList<>();
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)==' '){
                while(!stack.isEmpty()){
                    sb.append(stack.poll());
                }
                sb.append(" ");
            }else{
                stack.push(s.charAt(i));
            }
        }
        while(!stack.isEmpty()){
            sb.append(stack.poll());
        }

        return sb.toString();
    }

    int count=0;
    int result=0;
    public int kthSmallest(TreeNode root, int k) {
        travel(root,k);
        return result;
    }

    private void travel(TreeNode root,int k){
        if(root.left!=null)travel(root.left,k);
        count++;
        if(count==k) {
            result = root.val;
            return;
        }
        if(root.right!=null)travel(root.right,k);
    }
    public static void main(String[] args) {

        LeetCode lt=new LeetCode();

        System.out.println(lt.reverseWords("Let's take LeetCode contest"));

    }
}
