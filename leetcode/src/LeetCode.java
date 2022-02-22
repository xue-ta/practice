import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class LeetCode {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {}

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public LeetCode() {}

    // 1086. 前五科的均分
    public int[][] highFive(int[][] items) {
        Map<Integer, PriorityQueue<Integer>> m = new HashMap<>();
        for (int[] item : items) {
            PriorityQueue p =
                    m.getOrDefault(
                            item[0],
                            new PriorityQueue<>(
                                    100,
                                    new Comparator<Integer>() {
                                        @Override
                                        public int compare(Integer o1, Integer o2) {
                                            return o2 - o1;
                                        }
                                    }));
            p.add(item[1]);
            m.putIfAbsent(item[0], p);
        }
        int[][] result = new int[m.size()][2];

        int cnt = 0;
        for (Map.Entry<Integer, PriorityQueue<Integer>> e : m.entrySet()) {
            int sum = 0;
            for (int i = 0; i < 5; i++) {
                sum = sum + e.getValue().poll();
            }
            result[cnt][0] = e.getKey();
            result[cnt][1] = (sum / 5);
            cnt++;
        }

        return result;
    }

    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        LinkedList<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                while (!stack.isEmpty()) {
                    sb.append(stack.poll());
                }
                sb.append(" ");
            } else {
                stack.push(s.charAt(i));
            }
        }
        while (!stack.isEmpty()) {
            sb.append(stack.poll());
        }

        return sb.toString();
    }

    int count = 0;
    int result = 0;

    public int kthSmallest(TreeNode root, int k) {
        travel(root, k);
        return result;
    }

    private void travel(TreeNode root, int k) {
        if (root.left != null) travel(root.left, k);
        count++;
        if (count == k) {
            result = root.val;
            return;
        }
        if (root.right != null) travel(root.right, k);
    }

    public int coinChange(int[] coins, int amount) {
        int dp[][] = new int[coins.length][amount + 1];

        for (int j = 0; j <= amount; j++) {
            if (j % coins[0] == 0) {
                dp[0][j] = 1;
            } else {
                dp[0][j] = 0;
            }
        }

        for (int i = 0; i < coins.length; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i < coins.length; i++) {
            for (int j = 1; j <= amount; j++) {
                if (coins[i] <= j) {
                    dp[i][j] = dp[i][j - coins[i]] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[coins.length - 1][amount];
    }

    public void reverseString(char[] s) {
        int left = 0;
        int right = s.length - 1;
        while (left < right) {
            swap(s, left, right);
            left++;
            right--;
        }
    }

    void swap(char[] s, int left, int right) {
        char temp = s[left];
        s[left] = s[right];
        s[right] = temp;
    }

    public boolean canPartition(int[] nums) {
        int target = Arrays.stream(nums).sum();
        if (target % 2 != 0) return false;

        target = target / 2;

        int max = Arrays.stream(nums).max().getAsInt();
        if (max > target) return false;
        boolean[][] dp = new boolean[nums.length][target + 1];
        dp[0][nums[0]] = true;
        for (int i = 0; i < nums.length; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = 1; j <= target; j++) {
                if (nums[i] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i]];
                }
            }
        }

        return dp[nums.length - 1][target];
    }

    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public int jump(int[] nums) {
        int position = nums.length - 1;
        int steps = 0;
        while (position != 0) {
            for (int i = 0; i < position; i++) {
                if (nums[i] >= position - i) {
                    position = i;
                    steps++;
                    break;
                }
            }
        }
        return steps;
    }

    public void reorderList(ListNode head) {
        ListNode middle = findMiddle(head);

        ListNode head2 = reverseList(middle.next);
        middle.next = null;
        mergeList(head, head2);
    }

    private ListNode reverseList(ListNode listNode) {
        ListNode pre = null;
        ListNode cur = listNode;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    private ListNode findMiddle(ListNode listNode) {
        ListNode slow = listNode;
        ListNode fast = listNode;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public void mergeList(ListNode l1, ListNode l2) {
        ListNode l1_tmp;
        ListNode l2_tmp;
        while (l1 != null && l2 != null) {
            l1_tmp = l1.next;
            l2_tmp = l2.next;

            l1.next = l2;
            l1 = l1_tmp;

            l2.next = l1;
            l2 = l2_tmp;
        }
    }

    public int romanToInt(String s) {
        int sum = 0;
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);

        for (int i = 0; i < s.length(); i++) {
            if ((i + 1 < s.length() && (map.get(s.charAt(i + 1)) > map.get(s.charAt(i))))) {
                sum = sum - map.get(s.charAt(i));
            } else {
                sum = sum + map.get(s.charAt(i));
            }
        }
        return sum;
    }

    public int robotSim(int[] commands, int[][] obstacles) {
        int max = 0;
        Set<String> set = new HashSet<>();
        for (int[] obstacle : obstacles) {
            set.add(obstacle[0] + "," + obstacle[1]);
        }

        int[] curPosition = {0, 0};
        int[][] directs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int curDirectIndex = 0;
        for (int i = 0; i < commands.length; i++) {
            if (commands[i] == -1) {
                curDirectIndex = ((curDirectIndex + 1)) % 4;
            }
            if (commands[i] == -2) {
                curDirectIndex = ((curDirectIndex + 3)) % 4;
            }
            if (commands[i] > 0) {
                for (int j = 0; j < commands[i]; j++) {
                    int temp[] = new int[2];
                    temp[0] = directs[curDirectIndex][0] + curPosition[0];
                    temp[1] = directs[curDirectIndex][1] + curPosition[1];
                    if (set.contains(temp[0] + "," + temp[1])) {
                        break;
                    } else {
                        curPosition[0] = temp[0];
                        curPosition[1] = temp[1];
                        max = Math.max(max, curPosition[0] * curPosition[0] + curPosition[1] * curPosition[1]);
                    }
                }
            }
        }
        return max;
    }

    public int longestOnes(int[] nums, int k) {
        int left = 0;
        int right = 0;
        int zeros = 0;
        int max = 0;
        while (right < nums.length) {
            if (nums[right] == 0) {
                zeros++;
            }
            while (zeros > k) {
                if (nums[left] == 0) {
                    zeros--;
                }
                left++;
            }
            max = Math.max(max, right - left + 1);
            right++;
        }
        return max;
    }

    // 输入：records = "a(b(c)<3>d)<2>e"

    // 输出："abcccdbcccde"
    private String UnzipString(String records) {
        String result = "";

        LinkedList<StringBuilder> stack_res = new LinkedList<>();
        int cur_multi = 0;
        StringBuilder cur = new StringBuilder();
        stack_res.push(cur);
        for (char record : records.toCharArray()) {
            if (Character.isAlphabetic(record)) {
                stack_res.peek().append(record);
            }
            if (record == '(') {
                cur = new StringBuilder();
                stack_res.push(cur);
            }
            if (record == ')') {
                continue;
            }
            if (Character.isDigit(record)) {
                cur_multi = cur_multi * 10 + record - '0';
            }
            if (record == '<') {
                cur_multi = 0;
            }
            if (record == '>') {
                StringBuilder temp = stack_res.pop();
                for (int i = 0; i < cur_multi; i++) {
                    stack_res.peek().append(temp);
                }
            }
        }
        return stack_res.peek().toString();
    }

    // tasks = [1,3,2,4,6,5,0], mutexPairs = [[1,3],[4,5]]
    public int divideGroup(int[] tasks, int[][] mutexPairs) {
        int result = 1;

        Map<Integer, Set<Integer>> map = new HashMap<>();
        Arrays.stream(tasks).forEach(task -> map.put(task, new HashSet<>()));
        Arrays.stream(mutexPairs).forEach(ints -> map.get(ints[0]).add(ints[1]));

        Set<Integer> curSet = new HashSet<>();

        for (int i = 0; i < tasks.length; i++) {
            if (curSet.contains(tasks[i])) {
                result++;
                curSet.clear();
                curSet.addAll(map.get(tasks[i]));
            } else {
                curSet.addAll(map.get(tasks[i]));
            }
        }
        return result;
    }

    public String decodeString(String s) {
        LinkedList<StringBuilder> resStack = new LinkedList<>();
        LinkedList<Integer> multiStack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        int curMulti = 0;
        for (char c : s.toCharArray()) {
            if (Character.isAlphabetic(c)) {
                sb.append(c);
            }

            if (Character.isDigit(c)) {
                curMulti = curMulti * 10 + c - '0';
            }

            if (c == '[') {
                multiStack.push(curMulti);
                resStack.push(sb);
                sb = new StringBuilder();
                curMulti = 0;
            }

            if (c == ']') {
                int multiTemp = multiStack.pop();
                StringBuilder resTemp = resStack.pop();
                for (int i = 0; i < multiTemp; i++) resTemp.append(sb);
                sb = resTemp;
            }
        }
        return sb.toString();
    }

    public int[][] generate(int m, int n) {
        int[][] result = new int[m][n];
        Set<String> set = new HashSet<>();
        set.add(0 + "," + 0);
        set.add(Integer.toString(m - 1) + "," + Integer.toString(n - 1));
        Random random = new Random();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int temp = random.nextInt(9) + 1;
                result[i][j] = temp;
            }
        }
        for (int i = 0; i < 5; i++) {
            int c = random.nextInt(m);
            int r = random.nextInt(n);

            while (set.contains(c + ',' + r)) {
                c = random.nextInt(m);
                r = random.nextInt(n);
            }
            set.add(Integer.toString(c) + "," + Integer.toString(r));
            System.out.println(c + "," + r);
            result[c][r] = 'x';
        }
        return result;
    }

    public int minSubArrayLen(int target, int[] nums) {
        int min = nums.length;
        for (int r = 0, l = 0; r < nums.length; r++) {
            while (sum(nums, l, r) >= target) {
                min = Math.min(min, r - l + 1);
                l++;
            }
        }

        return sum(nums, 0, nums.length) < target ? 0 : min;
    }

    private int sum(int[] nums, int l, int r) {
        int sum = 0;
        while (l <= r) {
            sum = sum + nums[l];
            l++;
        }
        return sum;
    }

    public String reverseParentheses(String s) {
        LinkedList<StringBuilder> res_stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (Character.isAlphabetic(s.charAt(i))) {
                sb.append(s.charAt(i));
                continue;
            }
            if (s.charAt(i) == '(') {
                res_stack.push(sb);
                sb = new StringBuilder();
                continue;
            }
            if (s.charAt(i) == ')') {
                StringBuilder temp = res_stack.pop();
                sb = temp.append(sb.reverse());
            }
        }
        return sb.toString();
    }

    public int[] dailyTemperatures(int[] temperatures) {
        LinkedList<Integer> stack = new LinkedList<>();
        int[] result = new int[temperatures.length];
        for (int i = 0; i < temperatures.length; i++) {
            while ((!stack.isEmpty()) && (temperatures[i] > temperatures[stack.peek()])) {
                int index = stack.pop();
                result[index] = i - index;
            }
            stack.push(i);
        }
        return result;
    }


    int len = 0;

    public int islandPerimeter(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) dfs(grid, i, j);
            }
        }
        return len;
    }

    private void dfs(int[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j > grid[0].length || grid[i][j] == 1) {
            len++;
        } else {
            grid[i][j] = -1;
            dfs(grid, i + 1, j);
            dfs(grid, i - 1, j);
            dfs(grid, i, j + 1);
            dfs(grid, i, j - 1);
        }
    }

    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack();
        for (int ast : asteroids) {
            if (ast < 0) {
                while (!stack.isEmpty() && stack.peek() < Math.abs(ast) && stack.peek() > 0) {
                    stack.pop();
                    continue;
                }
                if (stack.isEmpty() || stack.peek() < 0) {
                    stack.push(ast);
                } else if (stack.peek() == -ast) {
                    stack.pop();
                }
            } else {
                stack.push(ast);
            }
        }

        int[] ans = new int[stack.size()];
        for (int t = ans.length - 1; t >= 0; --t) {
            ans[t] = stack.pop();
        }
        return ans;
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(target - nums[i])) {
                return new int[] {i, hashMap.get(target - nums[i])};
            }
            hashMap.put(nums[i], i);
        }
        return new int[] {};
    }

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    static class Solution {
        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
            return addTwoNumbers0(l1, l2, 0);
        }

        private ListNode addTwoNumbers0(ListNode l1, ListNode l2, int c) {
            if (l1 == null) return addTwoNumbers0(l2, c);
            if (l2 == null) return addTwoNumbers0(l1, c);
            ListNode root = new ListNode((l1.val + l2.val + c) % 10);
            root.next = addTwoNumbers0(l1.next, l2.next, (l1.val + l2.val + c) / 10);
            return root;
        }

        private ListNode addTwoNumbers0(ListNode l, int c) {
            if (l == null && c != 0) {
                return new ListNode(c);
            }
            if (l == null) return null;
            ListNode root = new ListNode((l.val + c) % 10);
            root.next = addTwoNumbers0(l.next, (l.val + c) / 10);
            return root;
        }
    }

    public int lengthOfLongestSubstring(String s) {
        int maxLen = 0, l = 0;
        HashMap<Character, Integer> map = new HashMap();
        for (int r = 0; r < s.length(); r++) {
            map.put(s.charAt(r), map.getOrDefault(s.charAt(r), 0) + 1);
            while (map.entrySet().size() < r - l + 1) {
                if (map.get(s.charAt(l)) > 1) {
                    map.put(s.charAt(l), map.getOrDefault(s.charAt(l), 0) - 1);
                } else {
                    map.remove(s.charAt(l));
                }
                l++;
            }
            maxLen = Math.max(maxLen, r - l + 1);
        }
        return maxLen;
    }

    public String longestPalindrome(String s) {
        char[] chars = s.toCharArray();
        int start = 0, end = 0;
        for (int i = 0; i < s.length() - 1; i++) {
            int len1 = findLong(i, i, chars);
            int len2 = findLong(i, i + 1, chars);
            if (Math.max(len1, len2) > end - start) {
                if (len1 > len2) {
                    start = i + (len - 1) / 2;
                    end = i + (len - 1) / 2;
                } else {
                    start = i + (len - 2) / 2;
                    end = i + 1 + (len - 2) / 2;
                }
            }
        }
        return s.substring(start, end + 1);
    }

    private int findLong(int l, int r, char[] chars) {
        while (l >= 0 && r < chars.length) {
            if (chars[l] == chars[r]) {
                l--;
                r++;
            } else break;
        }
        return r - l - 1;
    }


    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<String>();
        if (digits.length() == 0) {
            return combinations;
        }
        Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};

        backTrack(phoneMap,0,combinations,digits,new StringBuilder());
        return combinations;

    }

    private void backTrack(Map<Character,String> map,int length,List<String> result,String digits,StringBuilder temp){
        if(length==digits.length()){
            result.add(temp.toString());
            return;
        }
        for(int i=0;i<map.get(digits.charAt(length)).length();i++){
            temp.append(map.get(digits.charAt(length)).charAt(i));
            backTrack(map,length+1,result,digits,temp);
            temp.deleteCharAt(length);
        }
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode slow=head;
        ListNode fast=head;
        for(int i=0;i<n;i++){
            fast=fast.next;
        }
        while(fast.next!=null){
            fast=fast.next;
            slow=slow.next;
        }
        slow.next=slow.next.next;
        return head;
    }

    public boolean isValid(String s) {
        HashMap<Character,Character> hashMap=new HashMap<>();
        hashMap.put('}','{');
        hashMap.put(']','[');
        hashMap.put(')','(');
        LinkedList<Character> stack=new LinkedList<>();
        for(int i=0;i<s.length();i++){
            if(stack.isEmpty()){
                stack.push(s.charAt(i));
            }else if(hashMap.get(s.charAt(i))==stack.peek()){
                stack.pop();
            }else{
                stack.push(s.charAt(i));
            }
        }
        return stack.isEmpty();
    }
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if(list1==null) return list2;
        if(list2==null) return list1;
        if(list1.val<list2.val){
            list1.next=mergeTwoLists(list1.next,list2);
            return list1;
        }else{
            list2.next=mergeTwoLists(list1,list2.next);
            return list2;
        }
    }


    public List<String> result0=new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        backTrack0(new StringBuilder(),n,0,0);
        return result0;
    }

    private void backTrack0(StringBuilder tmp,int n,int left,int right){
        if(tmp.length()==2*n){
            result0.add(tmp.toString());
            return;
        }
        if(left<n){
            tmp.append("(");
            backTrack0(tmp,n,left+1,right);
            tmp.deleteCharAt(tmp.length()-1);
        }
        if(left>right){
            tmp.append(")");
            backTrack0(tmp,n,left,right+1);
            tmp.deleteCharAt(tmp.length()-1);
        }
    }

    List<List<Integer>> ret=new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        backTrack(nums,new ArrayList<>(),new int[nums.length]);
        return ret;
    }

    private void backTrack(int[] nums,List<Integer> temp,int[] mark){
        if(temp.size()==nums.length){
            ret.add(new ArrayList<>(temp));
        }
        for(int i=0;i<nums.length;i++){
            if(mark[i]==0) {
                mark[i]=1;
                temp.add(nums[i]);
                backTrack(nums,temp,mark);
                mark[i]=0;
                temp.remove(temp.size()-1);
            }
        }
    }



    public int trap(int[] height) {
        int sum = 0;

        int[] leftMax=new int[height.length];
        int[] rightMax=new int[height.length];
        for(int i=1;i< height.length;i++){
            leftMax[i]=Math.max(leftMax[i-1],height[i-1]);
        }
        for(int j= height.length-2;j>=0;j--){
            rightMax[j]=Math.max(leftMax[j+1],height[j+1]);
        }

        for(int i=1;i< height.length-1;i++){
            if(Math.min(leftMax[i],rightMax[i])>height[i]){
                sum=sum+Math.min(leftMax[i],rightMax[i])-height[i];
            };
        }
        return sum;
    }
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = new char[26];
            for (char c : str.toCharArray()) {
                chars[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < chars.length; i++) {
                if (chars[i] != 0) {
                    sb.append((char) ('a' + i));
                    sb.append(chars[i]);
                }
            }
            List<String> list = map.getOrDefault(sb.toString(), new ArrayList<String>());
            list.add(str);
            map.put(sb.toString(), list);
        }
        return new ArrayList<List<String>>(map.values());
    }

    public int maxSubArray(int[] nums) {
        int max=Integer.MIN_VALUE;
        int cursum=0;
        for(int i=0;i<nums.length;i++){
            if(cursum<0){
                cursum=nums[i];
            }else{
                cursum=cursum+nums[i];
            }
            max=Math.max(cursum,max);
        }
        return max;
    }

    public boolean canJump(int[] nums) {
        int max=1;
        for(int i=0;i<nums.length;i++){
            if(i<max){
                max=Math.max(max,i+nums[i]);
            }
            if(max>nums.length){
                return true;
            }
        }
        return false;
    }

    public int[][] merge(int[][] intervals) {
        List<int[]> merged=new ArrayList<>();
        Arrays.sort(intervals,(o1,o2)->o1[0]-o2[0]);

        for(int i=0;i<intervals.length;i++){
            int l=intervals[i][0],r=intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < l) {
                merged.add(new int[]{l, r});
            }else{
                merged.get(merged.size()-1)[1]=Math.max(merged.get(merged.size()-1)[1],r);
            }
        }

        return merged.toArray(new int[merged.size()-1][]);
    }

    public int uniquePaths(int m, int n) {
        int[][] dp=new int[m][n];
        for(int i=0;i<=m-1;i++){
            dp[i][0]=1;
        }
        for(int i=0;i<=n-1;i++){
            dp[0][i]=1;
        }
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                dp[i][j]=dp[i][j-1]+dp[i-1][j];
            }
        }
        return dp[m-1][n-1];
    }

    public int minPathSum(int[][] grid) {
        int m=grid.length,n=grid[0].length;
        int[][] dp=new int[m][n];
        int sum=0;
        for(int i=0;i<=m-1;i++){
            sum=grid[i][0]+sum;
            dp[i][0]=sum;
        }
        sum=0;
        for(int i=0;i<=n-1;i++){
            sum=grid[0][i]+sum;
            dp[0][i]=sum;
        }
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                dp[i][j]=Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }

    public int climbStairs(int n) {
        int[] dp=new int[n+1];
        dp[0]=0;
        dp[1]=1;
        for(int i=2;i<=n;i++){
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }

    public int minDistance(String word1, String word2) {

        if(word1.length()==0){
            return word2.length();
        }
        if(word2.length()==0){
            return word1.length();
        }
        int m=word1.length(),n=word2.length();
        int[][] dp=new int[m][n];
        for(int i=0;i<=m-1;i++){
            dp[i][0]=i;
        }
        for(int i=0;i<=n-1;i++){
            dp[0][i]=i;
        }
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(word1.charAt(i-1)==word2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1];
                }else {
                    dp[i][j] = Math.min(dp[i - 1][j-1]+1,Math.max(dp[i-1][j],dp[i][j-1])+1);
                }
            }
        }
        return dp[m-1][n-1];
    }


    public void sortColors(int[] nums) {
        int i=0,p0=0;
        int p2=nums.length-1;

        while(i<=p2){
            if(nums[i]==1){
                i++;
            }else if(nums[i]==2){
                swap(nums,i,p2);
                p2--;
            }else{
                swap(nums,i,p0);
                p0++;
                i++;
            }
        }
    }

    void swap(int[] s, int left, int right) {
        int temp = s[left];
        s[left] = s[right];
        s[right] = temp;
    }

    public String minWindow(String s, String t) {
        HashMap<Character,Integer> tmap=new HashMap<>();
        HashMap<Character,Integer> smap=new HashMap<>();
        int l=0,r=0;
        int start=-1,end=-1;
        int curMin=Integer.MAX_VALUE;
        for(char c:t.toCharArray()){
            tmap.put(c,tmap.getOrDefault(c,0)+1);
        }
        while(r<s.length()){
            smap.put(s.charAt(r),smap.getOrDefault(s.charAt(r),0)+1);

            while(check(smap,tmap)&&l<=r){
                if(r-l+1<curMin) {
                    start = l;
                    end = r;
                    curMin=end-start+1;
                }
                if (tmap.containsKey(s.charAt(l))) {
                    smap.put(s.charAt(l), smap.getOrDefault(s.charAt(l), 0) - 1);
                }
                l++;
            }
            r++;
        }
        if(start==-1&&end==-1){
            return "";
        }
        return s.substring(start,end+1);
    }

    private boolean check(Map<Character,Integer> s,Map<Character,Integer> t){
        return t.keySet().stream().allMatch(key->t.getOrDefault(key,0)<=(s.getOrDefault(key,0)));
    }



    List<List<Integer>> list=new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
        backTrack(nums,0,new ArrayList<>());
        return list;
    }

    private void backTrack(int[] nums,int index,List<Integer> temp){
        list.add(new ArrayList<>(temp));
        for(int i=index;i<nums.length;i++){
            temp.add(nums[i]);
            backTrack(nums,i+1,temp);
            temp.remove(temp.size()-1);
        }
    }



    int[][] direct=new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
    public boolean exist(char[][] board, String word) {
        int[][] mark=new int[board.length][board[0].length];
        for(int i=0;i< board.length;i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, i, j, 1, word,mark)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board,int i,int j,int index,String word,int[][] mark){
        if(word.charAt(index-1)!=board[i][j]){
            return false;
        }
        if(index==word.length()){
            return false;
        }
        mark[i][j]=1;
        int newi,newj;
        for(int[] dir:direct){
            newi=i+dir[0];
            newj=j+dir[1];
            if(newi>=0&&newi<= board.length-1&&newj>=0&&newj<=board[0].length){
                if(mark[newi][newj]==0){
                    if(dfs(board,newi,newj,index+1,word,mark)){
                        return true;
                    }
                }
                break;
            }
        }
        mark[i][j]=0;
        return false;
    }


    public int largestRectangleArea(int[] heights) {

        LinkedList<Integer> stack=new LinkedList<>();
        int max=0;
        for(int i=0;i< heights.length;i++){
            while(!stack.isEmpty()&&heights[stack.peek()]>heights[i]){
                max=Math.max(heights[stack.peek()]*(i-stack.peek()+1),max);
                stack.pop();
            }
            stack.push(i);
        }
        return max;
    }
    public static void main(String[] args) throws InterruptedException {
        LeetCode l=new LeetCode();
        l.largestRectangleArea(new int[]{2,4});
    }
}

class OddEven {
    private int n = 0;

    private volatile boolean odd = true;

    private String lock = new String("lock");

    public void odd() throws InterruptedException {
        while (n < 100) {
            synchronized (lock) {
                while (!odd) {
                    lock.wait();
                }
                print(n);
                n++;
                odd = false;
                lock.notify();
            }
        }
    }

    public void even() throws InterruptedException {
        while (n < 100) {
            synchronized (lock) {
                while (odd) {
                    lock.wait();
                }
                print(n);
                n++;
                odd = true;
                lock.notify();
            }
        }
    }

    private void print(int i) {
        System.out.println(i);
    }
}
