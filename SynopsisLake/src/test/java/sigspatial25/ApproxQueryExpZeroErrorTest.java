package sigspatial25;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class ApproxQueryExpZeroErrorTest {
    @Test
    public void zeroTruthDoesNotDisappearAsNanInAllQueryAnalysis() {
        assertEquals(0, ApproxQueryExp.relativeError(0, 0), 0);
        assertEquals(Double.POSITIVE_INFINITY, ApproxQueryExp.relativeError(1, 0), 0);
        assertEquals(.25, ApproxQueryExp.relativeError(125, 100), 0);
        assertEquals(.25, ApproxQueryExp.relativeError(75, 100), 0);
    }
}
