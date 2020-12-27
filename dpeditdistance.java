import java.util.*;

class DpEditDistance {

  static int min(int insert, int remove, int replace) {
    Integer[] values = { insert, remove, replace };
    return Collections.min(Arrays.asList(values));
  }

  static int calculateEditDistance(String str1, String str2) {

    int[][] memo = new int[str1.length() + 1][str2.length() + 1];

    for (int i = 0; i <= str1.length(); i++) {
      for (int j = 0; j <= str2.length(); j++) {
        if (i == 0) {
          memo[i][j] = j;
        }
        else if (j == 0) {
          memo[i][j] = i;
        }
        else if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
          memo[i][j] = memo[i - 1][j - 1];
        }
        else {
          memo[i][j] = min(1 + memo[i][j - 1],
                           1 + memo[i - 1][j],
                           2 + memo[i - 1][j - 1]);
        }
      }
    }

    return memo[str1.length()][str2.length()];
  }

  public static void main(String[] args) {
    String str1 = "intrinsic";
    String str2 = "intrusive";

    int minDistance = calculateEditDistance(str1, str2);
    System.out.println(minDistance);
  }
}
