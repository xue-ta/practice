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

    public int lengthOfLongestSubstring(String s) {
        int maxLen = 0, r = 0;
        Set<Character> set = new HashSet<>();
        for (int l = 0; l < s.length(); l++) {
            set.add(s.charAt(l));
            while (set.size() < l - r + 1) {
                if (s.charAt(r) != s.charAt(l)) {
                    set.remove(s.charAt(r));
                }
                r++;
            }
            maxLen = Math.max(maxLen, l - r + 1);
        }
        return maxLen;
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
        LinkedList<Integer> stack=new LinkedList<>();
        int[] result=new int[temperatures.length];
        for (int i=0;i< temperatures.length;i++){
            while((!stack.isEmpty())&&(temperatures[i]>temperatures[stack.peek()])){
                int index=stack.pop();
                result[index]=i-index;
            }
            stack.push(i);
        }
        return result;
    }

        public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
            if(l1==null) return l2;
            if(l2==null) return l1;
            if(l1.val<l2.val){
                l1.next=mergeTwoLists(l1.next,l2);
                return l1;
            }else{
                l2.next=mergeTwoLists(l1, l2.next);
                return l2;
            }
        }

    public static void main(String[] args) {
        LeetCode lt = new LeetCode();

        lt.dailyTemperatures(new int[]{73,74,75,71,69,72,76,73});

        List<String> l=new LinkedList<>();
        l.add("1,2,3,4");
        l.add("5,6,7,8");
        l.add("9");
        l.stream().flatMap(o-> Arrays.stream(o.split(","))).collect(Collectors.toList());


    }
}
