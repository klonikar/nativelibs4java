
package com.nativelibs4java.opencl;

import static com.nativelibs4java.opencl.JavaCL.createBestContext;
import static com.nativelibs4java.util.NIOUtils.directFloats;
import static org.junit.Assert.assertEquals;
import org.bridj.*;
import static org.bridj.Pointer.*;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import java.nio.FloatBuffer;

import org.junit.BeforeClass;
import org.junit.Test;

import com.nativelibs4java.test.MiscTestUtils;
import com.nativelibs4java.util.NIOUtils;

public class OpenCL4JavaBasicTest {

    public static final double ABSOLUTE_FLOAT_ERROR_TOLERANCE = 2e-4;
    public static final double RELATIVE_FLOAT_ERROR_TOLERANCE = 5e-8;

    @Test
    public void simpleTest() {
        try {
			CLContext context = createBestContext();
			
            int dataSize = 100000;
			String dataSizeStr = System.getProperty("dataSize", Integer.toString(dataSize));
			if(dataSizeStr != null && !dataSizeStr.isEmpty()) {
				try {
					dataSize = Integer.parseInt(dataSizeStr);
				} catch(Exception ex) {}
			}
			String src = "\n" +
                    "__kernel void aSinB(                                                   \n" +
                    "   __global const float* a,                                       \n" +
                    "   __global const float* b,                                       \n" +
                    "   __global float* output)                                        \n" +
                    "{                                                                 \n" +
                    "   int i = get_global_id(0);                                      \n" +
                    "   output[i] = a[i] * sin(b[i]) + 1;                              \n" +
                    "}                                                                 \n";

			//System.out.println("Kernel function: " +  src);
            CLProgram program = context.createProgram(src).build();
			CLKernel kernel = program.createKernel("aSinB");
            CLQueue queue = context.createDefaultQueue();

            /// Create direct NIO buffers and fill them with data in the correct byte order
            Pointer<Float> a = allocateFloats(dataSize).order(context.getKernelsDefaultByteOrder());
            Pointer<Float> b = allocateFloats(dataSize).order(context.getKernelsDefaultByteOrder());
            for (int i = 0; i < dataSize; i++) {
                float value = (float)i;
                a.set(i, value);
                b.set(i, value);
            }

            // Allocate OpenCL-hosted memory for inputs and output, 
            // with inputs initialized as copies of the NIO buffers
			long t1_g = System.currentTimeMillis();
            CLBuffer<Float> memIn1 = context.createBuffer(CLMem.Usage.Input, a, true); // 'true' : copy provided data
            CLBuffer<Float> memIn2 = context.createBuffer(CLMem.Usage.Input, b, true);
            CLBuffer<Float> memOut = context.createBuffer(CLMem.Usage.Output, Float.class, dataSize);

            // Bind these memory objects to the arguments of the kernel
            kernel.setArgs(memIn1, memIn2, memOut);

            // Ask for execution of the kernel with global size = dataSize
            //   and workgroup size = 1
            kernel.enqueueNDRange(queue, new int[]{dataSize});

            // Wait for all operations to be performed
            queue.finish();

            // Copy the OpenCL-hosted array back to RAM
            Pointer<Float> output = memOut.read(queue);
			long t2_g = System.currentTimeMillis();

			int numProcessors = Runtime.getRuntime().availableProcessors();
			int chunkSize = dataSize/numProcessors;
			System.out.println("number of processors/cores: " + numProcessors + ", CPU chunkSize: " + chunkSize);
			long t1 = System.currentTimeMillis();
			ExecutorService taskExecutor = Executors.newFixedThreadPool(numProcessors);
			MyRunnable[] tasks = new MyRunnable[numProcessors];
			for(int i = 0;i < numProcessors;i++) {
				MyRunnable task = new MyRunnable(i*chunkSize, (i+1)*chunkSize);
				tasks[i] = task;
				taskExecutor.execute(task);
			}
			taskExecutor.shutdown();
			try {
				taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch(Exception ex) { }
			long t2 = System.currentTimeMillis();
			System.out.println("gpu time diff: " + (t2_g-t1_g) + " ms, cpu time diff: " + (t2-t1) + " ms");
            // Compute absolute and relative average errors wrt Java implem
            double totalAbsoluteError = 0, totalRelativeError = 0;
			for(int i = 0;i < dataSize;i++) {
				float expected = tasks[i/chunkSize].expectedResults[i % chunkSize];
                float result = output.get(i);
                double d = result - expected;
                if (expected != 0) {
                    totalRelativeError += d / expected;
                }

                totalAbsoluteError += d < 0 ? -d : d;
            }
            double avgAbsoluteError = totalAbsoluteError / dataSize;
            double avgRelativeError = totalRelativeError / dataSize;
            System.out.println("Average absolute error = " + avgAbsoluteError);
            System.out.println("Average relative error = " + avgRelativeError);

            assertEquals("Bad relative error", 0, avgRelativeError, RELATIVE_FLOAT_ERROR_TOLERANCE);
            assertEquals("Bad absolute error", 0, avgAbsoluteError, ABSOLUTE_FLOAT_ERROR_TOLERANCE);

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
	
	private static class MyRunnable implements Runnable {
		private int start;
		private int end;
		public float[] expectedResults;

		public MyRunnable(int start, int end) {
			this.start = start;
			this.end = end;
			int dataSize = end - start + 1;
			expectedResults = new float[dataSize];
		}
		
		public void run() {
			for (int i = start; i < end; i++) {
                expectedResults[i-start] = i * (float) Math.sin(i) + 1;
			}
		}
	}
}
