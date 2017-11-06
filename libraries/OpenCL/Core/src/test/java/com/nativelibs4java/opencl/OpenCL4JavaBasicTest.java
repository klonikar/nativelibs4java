package trial.javacl;

import static com.nativelibs4java.opencl.JavaCL.createBestContext;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.security.SecureRandom;
import java.util.Date;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
// Download javacl jar from: https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/javacl/javacl-1.0.0-RC3-shaded.jar
// javac -cp javacl-1.0.0-RC3-shaded.jar -d . OpenCL4JavaBasicTest.java
// java -cp javacl-1.0.0-RC3-shaded.jar;. trial.javacl.OpenCL4JavaBasicTest 1000000
public class OpenCL4JavaBasicTest {
	private static class MyRunnable implements Runnable {
		private int start;
		private int end;
		public float[] expectedResults;
		float[] aVals;
		float[] bVals;
		public int chunkSize;

		public MyRunnable(int start, int end, float[] aVals, float[] bVals) {
			this.start = start;
			this.end = end;
			this.chunkSize = end - start;
			int dataSize = end - start + 1;
			this.aVals = aVals;
			this.bVals = bVals;
			expectedResults = new float[dataSize];
		}
		
		public void run() {
			for (int i = start; i < end; i++) {
				float a = aVals[i];
				float b = bVals[i];
                expectedResults[i-start] = (float) Math.sin(Math.exp(Math.cos(Math.sin(a) * Math.sin(b) + 1)));
			}
		}
	}

    private static MyRunnable[] executeOnHost(int dataSize, float[] aVals, float[] bVals) {
		int numProcessors = Runtime.getRuntime().availableProcessors();
		int chunkSize = dataSize/numProcessors;
		System.out.println("number of processors/cores: " + numProcessors + ", CPU chunkSize: " + chunkSize);
		long t1 = System.currentTimeMillis();
		ExecutorService taskExecutor = Executors.newFixedThreadPool(numProcessors);
		MyRunnable[] tasks = new MyRunnable[numProcessors];
		for(int i = 0;i < numProcessors;i++) {
			MyRunnable task = new MyRunnable(i*chunkSize, (i+1)*chunkSize, aVals, bVals);
			tasks[i] = task;
			taskExecutor.execute(task);
		}
		taskExecutor.shutdown();
		try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch(Exception ex) { }
		long t2 = System.currentTimeMillis();
		System.out.println("Host execute time: " + (t2-t1) + " ms");
		return tasks;
    }

	private static class MyRunnableDouble implements Runnable {
		private int start;
		private int end;
		public double[] expectedResults;
		double[] aVals;
		double[] bVals;
		public int chunkSize;

		public MyRunnableDouble(int start, int end, double[] aVals, double[] bVals) {
			this.start = start;
			this.end = end;
			this.chunkSize = end - start;
			int dataSize = end - start + 1;
			this.aVals = aVals;
			this.bVals = bVals;
			expectedResults = new double[dataSize];
		}
		
		public void run() {
			for (int i = start; i < end; i++) {
				double a = aVals[i];
				double b = bVals[i];
                expectedResults[i-start] = Math.sin(Math.exp(Math.cos(Math.sin(a) * Math.sin(b) + 1)));
			}
		}
	}

    private static MyRunnableDouble[] executeOnHostDouble(int dataSize, double[] aVals, double[] bVals) {
		int numProcessors = Runtime.getRuntime().availableProcessors();
		int chunkSize = dataSize/numProcessors;
		System.out.println("number of processors/cores: " + numProcessors + ", CPU chunkSize: " + chunkSize);
		long t1 = System.currentTimeMillis();
		ExecutorService taskExecutor = Executors.newFixedThreadPool(numProcessors);
		MyRunnableDouble[] tasks = new MyRunnableDouble[numProcessors];
		for(int i = 0;i < numProcessors;i++) {
			MyRunnableDouble task = new MyRunnableDouble(i*chunkSize, (i+1)*chunkSize, aVals, bVals);
			tasks[i] = task;
			taskExecutor.execute(task);
		}
		taskExecutor.shutdown();
		try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch(Exception ex) { }
		long t2 = System.currentTimeMillis();
		System.out.println("Host execute time: " + (t2-t1) + " ms");
		return tasks;
    }

    private static double[] computeDifference(Pointer<Float> output, MyRunnable[] tasks, int dataSize) {
        // Compute absolute and relative average errors wrt Java implem
        double totalAbsoluteError = 0, totalRelativeError = 0;
		for(int i = 0;i < dataSize;i++) {
			float expected = tasks[i/tasks[0].chunkSize].expectedResults[i % tasks[0].chunkSize];
            float result = output.get(i);
            double d = result - expected;
            if (expected != 0) {
                totalRelativeError += d / expected;
            }

            totalAbsoluteError += d < 0 ? -d : d;
        }
		output.release();
        double avgAbsoluteError = totalAbsoluteError / dataSize;
        double avgRelativeError = totalRelativeError / dataSize;
        return new double[] {avgAbsoluteError, avgRelativeError};
    }
    
    private static double[] computeDifferenceDouble(Pointer<Double> output, MyRunnableDouble[] tasks, int dataSize) {
        // Compute absolute and relative average errors wrt Java implem
        double totalAbsoluteError = 0, totalRelativeError = 0;
		for(int i = 0;i < dataSize;i++) {
			double expected = tasks[i/tasks[0].chunkSize].expectedResults[i % tasks[0].chunkSize];
            double result = output.get(i);
            double d = result - expected;
            if (expected != 0) {
                totalRelativeError += d / expected;
            }

            totalAbsoluteError += d < 0 ? -d : d;
        }
		output.release();
        double avgAbsoluteError = totalAbsoluteError / dataSize;
        double avgRelativeError = totalRelativeError / dataSize;
        return new double[] {avgAbsoluteError, avgRelativeError};
    }

    private static Pointer<Double> executeOnDeviceDouble(CLKernel kernel, CLContext context, int dataSize, int blockSize, double[] aVals, double[] bVals) {
        CLQueue queue = context.createDefaultQueue();
        // Ask for execution of the kernel with global size = dataSize
        int numThreads = ((dataSize-1)/blockSize + 1)*blockSize;
		System.out.println("dataSize: " + dataSize + ", numThreads: " + numThreads + ", blockSize: " + blockSize);

        long t1_g = System.currentTimeMillis();
        /// Create direct NIO buffers and fill them with data in the correct byte order
        //Pointer<Double> a = pointerToDoubles(aVals).order(context.getKernelsDefaultByteOrder());
        //Pointer<Double> b = pointerToDoubles(bVals).order(context.getKernelsDefaultByteOrder());

        // Allocate OpenCL-hosted memory for inputs and output, 
        // with inputs initialized as copies of the NIO buffers
        DoubleBuffer aBuffer = DoubleBuffer.wrap(aVals);
        DoubleBuffer bBuffer = DoubleBuffer.wrap(bVals);
        CLBuffer<Double> memIn1 = context.createDoubleBuffer(CLMem.Usage.Input, aBuffer, true); // 'true' : copy provided data
        CLBuffer<Double> memIn2 = context.createDoubleBuffer(CLMem.Usage.Input, bBuffer, true);
        CLBuffer<Double> memOut = context.createBuffer(CLMem.Usage.Output, Double.class, dataSize);
		long t_dataXfr1_g = System.currentTimeMillis();
        // Bind these memory objects to the arguments of the kernel
        kernel.setArgs(memIn1, memIn2, memOut, dataSize);

        kernel.enqueueNDRange(queue, new int[]{numThreads}, new int[]{blockSize});

        // Wait for all operations to be performed
        queue.finish();
		long t_execute_g = System.currentTimeMillis();
		
        // Copy the OpenCL-hosted array back to RAM
        Pointer<Double> output = memOut.read(queue);
		long t2_g = System.currentTimeMillis();
		memIn1.release();
		memIn2.release();
		System.out.println("Device data transfer time: " + ((t_dataXfr1_g - t1_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr1_g) + "ms");
		System.out.println("Device time diff: " + (t2_g-t1_g) + " ms");

		return output;
    }

    private static Pointer<Float> executeOnDevice(CLKernel kernel, CLContext context, int dataSize, int blockSize, float[] aVals, float[] bVals) {
        CLQueue queue = context.createDefaultQueue();
        // Ask for execution of the kernel with global size = dataSize
        int numThreads = ((dataSize-1)/blockSize + 1)*blockSize;
		System.out.println("dataSize: " + dataSize + ", numThreads: " + numThreads + ", blockSize: " + blockSize);

		long t1_g = System.currentTimeMillis();
        /// Create direct NIO buffers and fill them with data in the correct byte order
        //Pointer<Float> a = pointerToFloats(aVals).order(context.getKernelsDefaultByteOrder());
        //Pointer<Float> b = pointerToFloats(bVals).order(context.getKernelsDefaultByteOrder());

        // Allocate OpenCL-hosted memory for inputs and output, 
        // with inputs initialized as copies of the NIO buffers
        FloatBuffer aBuffer = FloatBuffer.wrap(aVals);
        FloatBuffer bBuffer = FloatBuffer.wrap(bVals);
        CLBuffer<Float> memIn1 = context.createFloatBuffer(CLMem.Usage.Input, aBuffer, true); // 'true' : copy provided data
        CLBuffer<Float> memIn2 = context.createFloatBuffer(CLMem.Usage.Input, bBuffer, true);
        CLBuffer<Float> memOut = context.createBuffer(CLMem.Usage.Output, Float.class, dataSize);
		long t_dataXfr1_g = System.currentTimeMillis();
        // Bind these memory objects to the arguments of the kernel
        kernel.setArgs(memIn1, memIn2, memOut, dataSize);

        kernel.enqueueNDRange(queue, new int[]{numThreads}, new int[]{blockSize});

        // Wait for all operations to be performed
        queue.finish();
		long t_execute_g = System.currentTimeMillis();
		
        // Copy the OpenCL-hosted array back to RAM
        Pointer<Float> output = memOut.read(queue);
		long t2_g = System.currentTimeMillis();
		memIn1.release();
		memIn2.release();
		System.out.println("Device data transfer time: " + ((t_dataXfr1_g - t1_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr1_g) + "ms");
		System.out.println("Device time diff: " + (t2_g-t1_g) + " ms");

		return output;
    }

    private static Pointer<Float> executeOnDeviceDist(CLKernel kernel, CLContext context, int blockSize, int num_vectors, float[] aVals, float[] bVals) {
        CLQueue queue = context.createDefaultQueue();
        int numThreads = num_vectors*blockSize;
		System.out.println("num_vectors: " + num_vectors + ", numThreads: " + numThreads + ", blockSize: " + blockSize);

		float[] a1Vals = new float[aVals.length];
		for(int i = 0;i < a1Vals.length;i++)
			a1Vals[i] = aVals[i] + 0.5f;
		
		long t1_g = System.currentTimeMillis();
        /// Create direct NIO buffers and fill them with data in the correct byte order
        //Pointer<Float> a = pointerToFloats(aVals).order(context.getKernelsDefaultByteOrder());
        //Pointer<Float> b = pointerToFloats(bVals).order(context.getKernelsDefaultByteOrder());

        // Allocate OpenCL-hosted memory for inputs and output, 
        // with inputs initialized as copies of the NIO buffers
        //FloatBuffer aBuffer = FloatBuffer.wrap(aVals);
        FloatBuffer bBuffer = FloatBuffer.wrap(bVals);
        CLBuffer<Float> memIn1 = context.createFloatBuffer(CLMem.Usage.Input, aVals.length);
        CLBuffer<Float> memIn2 = context.createFloatBuffer(CLMem.Usage.Input, bBuffer, true); // 'true' : copy provided data
        CLBuffer<Float> memOut = context.createBuffer(CLMem.Usage.Output, Float.class, num_vectors);
		long t_dataXfr1_g = System.currentTimeMillis();
		System.out.println("Device data transfer time for database buffer: " + (t_dataXfr1_g - t1_g) + "ms");
        // Bind these memory objects to the arguments of the kernel
        kernel.setArgs(memIn1, memIn2, memOut);
        kernel.setLocalArg(3, aVals.length*4); // size in bytes to fit one vector length worth of floats

        Pointer<Float> output = null;
        for(int i = 0;i < 10;i++) {  // execute the kernel number of times for performance test. Return output of final run.
        	long t_dataXfr2_g = System.currentTimeMillis();
	        Pointer<Float> aIn = Pointer.pointerToFloats(i % 2 != 0 ? aVals : a1Vals);
	        memIn1.write(queue, aIn, false);
	        long t_dataXfr3_g = System.currentTimeMillis();
	        kernel.enqueueNDRange(queue, new int[]{numThreads}, new int[]{blockSize});
	
	        // Wait for all operations to be performed
	        queue.finish();
			long t_execute_g = System.currentTimeMillis();
			
	        // Copy the OpenCL-hosted array back to RAM
	        output = memOut.read(queue);
	        
			long t2_g = System.currentTimeMillis();
	
			System.out.println("Device data transfer time: " + ((t_dataXfr3_g - t_dataXfr2_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr3_g) + "ms");
			System.out.println("Device time diff: " + (t2_g-t_dataXfr2_g) + " ms");
        }
        
		memIn1.release();
		memIn2.release();
		memOut.release();

		return output;
    }

    private static Pointer<Float> executeOnDeviceDist2(CLKernel kernel, CLContext context, int blockSize, int num_vectors, float[] aVals, float[] bVals) {
        CLQueue queue = context.createDefaultQueue();
        int numThreads = num_vectors;
		System.out.println("num_vectors: " + num_vectors + ", numThreads: " + numThreads + ", blockSize: " + blockSize);

		float[] a1Vals = new float[aVals.length];
		for(int i = 0;i < a1Vals.length;i++)
			a1Vals[i] = aVals[i] + 0.5f;

		long t1_g = System.currentTimeMillis();
        /// Create direct NIO buffers and fill them with data in the correct byte order
        //Pointer<Float> a = pointerToFloats(aVals).order(context.getKernelsDefaultByteOrder());
        //Pointer<Float> b = pointerToFloats(bVals).order(context.getKernelsDefaultByteOrder());

        // Allocate OpenCL-hosted memory for inputs and output, 
        // with inputs initialized as copies of the NIO buffers
        //FloatBuffer aBuffer = FloatBuffer.wrap(aVals);
        FloatBuffer bBuffer = FloatBuffer.wrap(bVals);
        CLBuffer<Float> memIn1 = context.createFloatBuffer(CLMem.Usage.Input, aVals.length);
        CLBuffer<Float> memIn2 = context.createFloatBuffer(CLMem.Usage.Input, bBuffer, true); // 'true' : copy provided data
        CLBuffer<Float> memOut = context.createBuffer(CLMem.Usage.Output, Float.class, num_vectors);
		long t_dataXfr1_g = System.currentTimeMillis();
		System.out.println("Device data transfer time for database buffer: " + (t_dataXfr1_g - t1_g) + "ms");
        // Bind these memory objects to the arguments of the kernel
        kernel.setArgs(memIn1, memIn2, memOut);

        Pointer<Float> output = null;
        for(int i = 0;i < 10;i++) {  // execute the kernel number of times for performance test. Return output of final run.
        	long t_dataXfr2_g = System.currentTimeMillis();
	        Pointer<Float> aIn = Pointer.pointerToFloats(i % 2 != 0 ? aVals : a1Vals);
	        memIn1.write(queue, aIn, false);
	        long t_dataXfr3_g = System.currentTimeMillis();
	        kernel.enqueueNDRange(queue, new int[]{numThreads}, new int[]{blockSize});
	
	        // Wait for all operations to be performed
	        queue.finish();
			long t_execute_g = System.currentTimeMillis();
			
	        // Copy the OpenCL-hosted array back to RAM
	        output = memOut.read(queue);
	        
			long t2_g = System.currentTimeMillis();
	
			System.out.println("Device data transfer time: " + ((t_dataXfr3_g - t_dataXfr2_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr3_g) + "ms");
			System.out.println("Device time diff: " + (t2_g-t_dataXfr2_g) + " ms");
        }
        
		memIn1.release();
		memIn2.release();
		memOut.release();

		return output;
    }

    // getting, setting and clearing bitMap array bits based on position(offset)
    private static void set(byte[] BitMap, int offset) 
    { 
           BitMap[offset >> 3]|= 1<<(offset & 7); 
    } 
    private static void clear(byte[] BitMap, int offset)
    { 
           BitMap[offset >> 3] &= ~(1 << (offset & 7)); 
    } 
    private boolean get(byte BitMap, int offset)
    {
    	return ((BitMap >> (offset & 7)) & 1) != 0;
    }

    //Kudu equivalent CPU implementation...just for testing purpose 
    private static void predicate_eval(Object col, int typeSize, long size, long lk, long hk,
    		 boolean lkValid, boolean hkValid, byte[] hostBitMap, boolean is_nullable, 
    		 byte[] null_bitmap)
    {
    	long colValue = 0;
    	if(col == null || hostBitMap == null || (is_nullable && (null_bitmap == null)))
    	{
    		return;	
    	}
    	else
    	{
    		for(int i=0; i<size; i++)
    		{

    			if (is_nullable && (null_bitmap[i >> 3] & (1 << (i & 7))) == 0)
    			{
    				clear(hostBitMap ,i);
    			}
    			else
    			{
    				if(typeSize == 4)
    					colValue = ((int[]) col)[i];
    				else if(typeSize == 8)
    					colValue = ((long[]) col)[i];
    				else
    					colValue = ((byte[]) col)[i];
    				if((lkValid && (colValue < lk)) || (hkValid && (colValue > hk)))
    				{
    					clear(hostBitMap ,i);// bitVec[i] = 0;
    				}
    				else
    				{
    					set(hostBitMap, i);
    				}
    			}
    		}
    	}
    }

    private static Pointer<Byte> executeOnDeviceFilter(CLKernel kernel, CLContext context, int blockSize, boolean isWithoutDivergence, Object colData, long dataSize, long lk, long hk,
   		                                               boolean lkValid, boolean hkValid, boolean is_nullable, 
   		                                               byte[] null_bitmap) {
        CLQueue queue = context.createDefaultQueue();
        // Ask for execution of the kernel with global size = dataSize
        int numThreads = 0;
        
        if(isWithoutDivergence)
        	numThreads = (int) (((dataSize-1)/(blockSize*8) + 1)*blockSize);
        else
        	numThreads = (int) (((dataSize-1)/(blockSize) + 1)*blockSize);
        
		System.out.println("dataSize: " + dataSize + ", bitmap size: " + null_bitmap.length + ", numThreads: " + numThreads + ", blockSize: " + blockSize);

		long t1_g = System.currentTimeMillis();

        // Allocate OpenCL-hosted memory for inputs and output, 
        // with inputs initialized as copies of the NIO buffers
        IntBuffer colBuffer = IntBuffer.wrap((int[]) colData);
        ByteBuffer null_bitmapBuffer = ByteBuffer.wrap(null_bitmap);
        CLBuffer<Integer> memIn1 = context.createIntBuffer(CLMem.Usage.Input, colBuffer, true); // 'true' : copy provided data
        CLBuffer<Byte> memIn2 = context.createByteBuffer(CLMem.Usage.Input, null_bitmapBuffer, true);
        CLBuffer<Byte> memOut = context.createBuffer(CLMem.Usage.Output, Byte.class, null_bitmap.length);
		long t_dataXfr1_g = System.currentTimeMillis();
        // Bind these memory objects to the arguments of the kernel
		int iLkValid = lkValid ? 1 : 0, iHkValid = hkValid ? 1 : 0, iIs_nullable = is_nullable ? 1 : 0;
        kernel.setArgs(memIn1, dataSize, lk, hk, iLkValid, iHkValid, memOut, iIs_nullable, memIn2);

        kernel.enqueueNDRange(queue, new int[]{numThreads}, new int[]{blockSize});

        // Wait for all operations to be performed
        queue.finish();
		long t_execute_g = System.currentTimeMillis();
		
        // Copy the OpenCL-hosted array back to RAM
        Pointer<Byte> output = memOut.read(queue);
		long t2_g = System.currentTimeMillis();
		memIn1.release();
		memIn2.release();
		System.out.println("Device data transfer time: " + ((t_dataXfr1_g - t1_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr1_g) + "ms");
		System.out.println("Device time diff: " + (t2_g-t1_g) + " ms");

		return output;
    }

    private static double[] computeDifferenceFilterResults(Pointer<Byte> output, byte[] cpuResults) {
        // Compute absolute and relative average errors wrt Java implementation
        double totalAbsoluteError = 0, totalRelativeError = 0;
		for(int i = 0;i < cpuResults.length;i++) {
			byte expected = cpuResults[i];
            byte result = output.get(i);
            double d = result - expected;
            //if(result != expected)
            //	System.out.format("Result %d different from expected %d at index: %d\n", result, expected, i);
            if (expected != 0) {
                totalRelativeError += d / expected;
            }

            totalAbsoluteError += d < 0 ? -d : d;
        }
		output.release();
        double avgAbsoluteError = totalAbsoluteError / cpuResults.length;
        double avgRelativeError = totalRelativeError / cpuResults.length;
        return new double[] {avgAbsoluteError, avgRelativeError};
    }

    public static void main(String[] args) {
        try {
        	CLContext context = createBestContext(DeviceFeature.GPU, DeviceFeature.MaxComputeUnits, DeviceFeature.DoubleSupport);;
			System.out.println("context: " + context);

        	CLPlatform[] platforms = JavaCL.listGPUPoweredPlatforms();
        	for(CLPlatform p : platforms) {
        		System.out.println("Platform: " + p.getName());
        		CLDevice[] devices = p.listGPUDevices(true);
        		for(CLDevice d : devices) {
        			System.out.println("Device: " + d.getName() + ", max compute units " + d.getMaxComputeUnits());
        			if(p.getName().toUpperCase().contains("NVIDIA") && d.getName().toUpperCase().contains("TITAN")) {
        				context.release();
        				context = d.getPlatform().createContext(null, new CLDevice[] { d });
        				break;
        			}
        			else if(p.getName().toUpperCase().contains("NVIDIA")) {
        				context.release();
        				context = d.getPlatform().createContext(null, new CLDevice[] { d });
        			}
        		}
        	}
        	System.out.println("new context: " + context);
        	
            int dataSize = 100000;
			String dataSizeStr = System.getProperty("dataSize", Integer.toString(dataSize));
			if(args != null && args.length > 0 && !args[0].isEmpty())
				dataSizeStr = args[0];

			try {
				dataSize = Integer.parseInt(dataSizeStr);
			} catch(Exception ex) {}
			
			boolean doubleMode = false;
			if(args != null && args.length > 1 && args[1].equalsIgnoreCase("double")) {
				doubleMode = true;
				System.out.println("Running in double mode...");
			}

			int blockSize = (int) context.getDevices()[0].getMaxWorkGroupSize();
			if(args != null && args.length > 2) {
				blockSize = Integer.parseInt(args[2]);
			}
			
			String src = "\n" +
					"#ifdef CONFIG_USE_DOUBLE\n"+
					"#if defined(cl_khr_fp64)  // Khronos extension available?\n"+
					"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"+
					"#define DOUBLE_SUPPORT_AVAILABLE\n"+
					"#elif defined(cl_amd_fp64)  // AMD extension available?\n"+
					"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"+
					"#define DOUBLE_SUPPORT_AVAILABLE\n"+
					"#endif\n"+
					"#endif // CONFIG_USE_DOUBLE\n"+

					"#if defined(DOUBLE_SUPPORT_AVAILABLE)\n"+

					"// double\n"+
					"typedef double real_t;\n"+
					"typedef double2 real2_t;\n"+
					"typedef double3 real3_t;\n"+
					"typedef double4 real4_t;\n"+
					"typedef double8 real8_t;\n"+
					"typedef double16 real16_t;\n"+
					"#define PI 3.14159265358979323846\n"+

					"#else\n"+

					"// float\n"+
					"typedef float real_t;\n"+
					"typedef float2 real2_t;\n"+
					"typedef float3 real3_t;\n"+
					"typedef float4 real4_t;\n"+
					"typedef float8 real8_t;\n"+
					"typedef float16 real16_t;\n"+
					"#define PI 3.14159265359f\n"+

					"#endif\n"+

                    "__kernel void aSinB(                                                   \n" +
                    "   __global const real_t* a,                                       \n" +
                    "   __global const real_t* b,                                       \n" +
                    "   __global real_t* output, int dataSize)                           \n" +
                    "{                                                                 \n" +
                    "   int i = get_global_id(0);                                      \n" +
                    "   if(i < dataSize) output[i] = sin(exp(cos(sin(a[i]) * sin(b[i]) + 1)));                              \n" +
                    "}                                                                 \n";
			
			//System.out.println("Kernel function: " +  src);
            CLProgram program = context.createProgram(src);
            if(doubleMode)
            	program = program.defineMacro("CONFIG_USE_DOUBLE", "1");
            
            program = program.build();
			CLKernel kernel = program.createKernel("aSinB");
            System.out.println("kernel workgroup size: " + kernel.getWorkGroupSize());
            for(CLDevice d : context.getDevices()) {
                long[] sizes = d.getMaxWorkItemSizes();
                System.out.println("Device: " + d.getName() + ", max workgroup size: " + d.getMaxWorkGroupSize()
                                                            + ", workItemSizes: " + sizes[0] + ", " + sizes[1] + ", " + sizes[2]);
            }

			if(doubleMode) {
		        double[] aVals = new double[dataSize];
		        double[] bVals = new double[dataSize];
		        for (int i = 0; i < dataSize; i++) {
		            double value = (double)i;
		            //a.set(i, value);
		            //b.set(i, value);
		            aVals[i] = value;
		            bVals[i] = value;
		        }
/*
				Pointer<Double> output = executeOnDeviceDouble(kernel, context, dataSize, blockSize, aVals, bVals);
				MyRunnableDouble[] tasks = executeOnHostDouble(dataSize, aVals, bVals);
				double[] diff = computeDifferenceDouble(output, tasks,  dataSize);
	            System.out.println("Average absolute error = " + diff[0]);
	            System.out.println("Average relative error = " + diff[1]);				
*/			}
			else {
		        float[] aVals = new float[dataSize];
		        float[] bVals = new float[dataSize];
		        for (int i = 0; i < dataSize; i++) {
		            float value = (float)i;
		            //a.set(i, value);
		            //b.set(i, value);
		            aVals[i] = value;
		            bVals[i] = value;
		        }

/*				Pointer<Float> output = executeOnDevice(kernel, context, dataSize, blockSize, aVals, bVals);
				MyRunnable[] tasks = executeOnHost(dataSize, aVals, bVals);
				double[] diff = computeDifference(output, tasks,  dataSize);
	            System.out.println("Average absolute error = " + diff[0]);
	            System.out.println("Average relative error = " + diff[1]);
*/			}
	
			/*
			String structured_src = "\n" +
			"typedef struct trial_aparapi_AparapiTrial$MyStruct_s{\n" +
			"float  a;\n" + 
			"float  b;\n" +
			"float  output;\n" +
   			"} trial_aparapi_AparapiTrial$MyStruct;\n" +
			"typedef struct This_s{\n" +
			"   __global trial_aparapi_AparapiTrial$MyStruct *s;\n" +
			"   int passid;\n" +
			"}This;\n" +
			"int get_pass_id(This *this){\n" +
			"   return this->passid;\n" +
			"}\n" +
			"__kernel void run(\n" +
			"   __global trial_aparapi_AparapiTrial$MyStruct *s, \n" +
			"   int passid\n" +
			"){\n" +
			"   This thisStruct;\n" +
			"   This* this=&thisStruct;\n" +
			"   this->s = s;\n" +
			"   this->passid = passid;\n" +
			"   {\n" +
			"	  int i = get_global_id(0);\n" +
			"	  this->s[i].output=sin(exp(cos(((sin(this->s[i].a) * sin(this->s[i].b)) + 1.0f))));\n" +
			"	  return;\n" +
			"   }\n" +
			"}\n" +
			"";

			String filterPredicateKernel = "\n" +
					" typedef unsigned char uchar_t; \n" +
					"__kernel void transform_no_divergence(global TYPE *col, long size, long lk, long hk,   \n" +
					"         int lkValid, int hkValid, global uchar_t *bitVec, int is_nullable, global const uchar_t *null_bitmap) \n" +
					"{ \n" +
					"  int gloId = get_global_id(0); \n" +
					"  int bitmapSize = (size - 1)/8 + 1; \n" +
					"  //if(gloId < bitmapSize) { \n" +
					"    //int offset = get_global_offset(0); \n" +
					"    int locId = get_local_id(0); \n" +
					"    int groupId = get_group_id(0); \n" +
					"    int localSize = get_local_size(0); \n" +
					"    int locId8 = locId << 3; \n" +
					"    uchar_t finalByte = 0; \n" +
					"    uchar_t null_bits = null_bitmap[gloId]; \n" +
					"    int startIndex = gloId + 7*groupId*localSize; \n" +
					"    global TYPE *colValuesP = col + startIndex; \n" +
					"    local TYPE localVals[LOCAL_WORK_SIZE << 3]; \n" +
					"    // Get values in local shared memory in a coalesced manner. One thread gets 8 values. \n" +
					"    for(int i = 0;i < 8;i++) { \n" +
					"        int iBlock = i*localSize; \n" +
					"        if((startIndex + iBlock) < size) \n" +
					"            localVals[locId + iBlock] = colValuesP[iBlock]; \n" +
					"        else \n" +
					"            localVals[locId + iBlock] = (TYPE) 0; \n" +
					"    } \n" + 
					"    barrier(CLK_LOCAL_MEM_FENCE); \n " +
					"    for(int i = 0; i < 8;i++) { \n" +
					"        bool null_check = is_nullable && !(null_bits & (1 << i)); \n" +
					"        bool filter_check = (lkValid && localVals[locId8 + i] < lk) || (hkValid && localVals[locId8 + i] > hk); \n" +
					"        bool check = !null_check && !filter_check; \n" +
					"        if(check) \n" +
					"            finalByte |= 1 << i; \n" +
					"    } \n" +
					"    // This ensures that we access global memory for this byte only once. \n" +
					"    bitVec[gloId] = finalByte; \n" +
					"  //} \n" +
					"} \n" +
				    "\n" +
					"__kernel void transform(global TYPE *col, long size, long lk, long hk,   \n" +
					"         int lkValid, int hkValid, global uchar_t *bitVec, int is_nullable, global const uchar_t *null_bitmap) \n" +
					"{ \n" +
					"    int gloId = get_global_id(0); \n" +
					"    if(gloId < size) { \n" +
					"        TYPE colValue = col[gloId]; \n" +
					"        bool null_check = is_nullable && !(null_bitmap[gloId >> 3] & (1 << (gloId & 7))); \n" +
					"        if(!(gloId & 7)) \n" +
					"            bitVec[gloId >> 3] = 0; \n" +
					"        bool filter_check = (lkValid && (colValue < lk)) || (hkValid && (colValue > hk)); \n" +
					"        bool check = !null_check && !filter_check; \n" +
					"        for(int i = 0; i < 8; i++) \n" +
					"        { \n" +
					"            if((gloId & 7) == i) \n" +
					"            {        \n" +
					"                 if(check) \n" +
					"                 { \n" +
					"                     bitVec[gloId >> 3] |= 1 << i; \n" +
					"                 } \n" +
					"            } \n" +
					"            barrier(CLK_GLOBAL_MEM_FENCE); \n" +
					"        } \n" +
					"    } \n" +
					"} \n"
			;

			int typeSize = 4;
			blockSize = 256;
			
			CLProgram programFilterKernel = context.createProgram(filterPredicateKernel);
            programFilterKernel = programFilterKernel.defineMacro("LOCAL_WORK_SIZE", Integer.toString(blockSize));
            if(typeSize == 4)
            	programFilterKernel = programFilterKernel.defineMacro("TYPE", "int");
            else if(typeSize == 8)
            	programFilterKernel = programFilterKernel.defineMacro("TYPE", "long");
            else
            	programFilterKernel = programFilterKernel.defineMacro("TYPE", "uchar_t");
            
            programFilterKernel = programFilterKernel.build();
            
			CLKernel kernelFilterKernel_no_divergence = programFilterKernel.createKernel("transform_no_divergence");
			
			CLKernel kernelFilterKernel_with_divergence = programFilterKernel.createKernel("transform");

			dataSize = 16*1024*1024;
	        int[] col = new int[dataSize];
			int bitmapSize = (dataSize - 1)/8 + 1;
	        byte[] null_bitmap = new byte[bitmapSize];
	        byte[] hostResults = new byte[bitmapSize];
	        Random colGenerator = new SecureRandom();
	        Random nullGenerator = new SecureRandom();
	        long lk = 500, hk = 100000;
	        boolean lkValid = true, hkValid = true, is_nullable = true;

	        colGenerator.setSeed(new Date().getTime());
	        nullGenerator.setSeed(new Date().getTime());

	        for (int i = 0; i < dataSize; i++) {
	            col[i] = colGenerator.nextInt((int) (2*hk));
	        }
	        for (int i = 0; i < bitmapSize; i++) {
	            null_bitmap[i] = (byte) nullGenerator.nextInt(256);
	        }

			predicate_eval(col, typeSize, dataSize, lk, hk, lkValid, hkValid, hostResults, is_nullable, null_bitmap);

			System.out.println("*** Filter computation without thread divergence ****");
			for(int i = 0;i < 10;i++)
			{
		        Pointer<Byte> output = executeOnDeviceFilter(kernelFilterKernel_no_divergence, context, blockSize, true, col, dataSize, lk, hk, lkValid, hkValid, is_nullable, null_bitmap);
		        double[] diff = computeDifferenceFilterResults(output, hostResults);
	            System.out.println("Kernel without divergence: Average absolute error = " + diff[0] + ", Average relative error = " + diff[1]);
			}			

			System.out.println("*** Filter computation with thread divergence ****");
			for(int i = 0;i < 10;i++)
			{
				Pointer<Byte> output = executeOnDeviceFilter(kernelFilterKernel_with_divergence, context, blockSize, false, col, dataSize, lk, hk, lkValid, hkValid, is_nullable, null_bitmap);
	            double[] diff = computeDifferenceFilterResults(output, hostResults);
	            System.out.println("Kernel with divergence: Average absolute error = " + diff[0] + ", Average relative error = " + diff[1]);
			}
			*/
	        
	        String distKernels = 
	        		"#define FOUR_ONES ((float4) (1.0f)) \n" +
	        		"__kernel void dist(__constant REAL_TYPE* a_vec, __global REAL_TYPE* b_vec,\n" + 
	        		"      __global float* output, __local REAL_TYPE* partial_dot) {\n" + 
	        		"\n" + 
	        		"   int gid = get_global_id(0);\n" + 
	        		"   int lid = get_local_id(0);\n" + 
	        		"   int grId = get_group_id(0);\n" + 
	        		"   int group_size = get_local_size(0);\n" + 
	        		"   REAL_TYPE sum = (REAL_TYPE) (0.0f);\n" + 
	        		"\n" + 
	        		"   REAL_TYPE diff = a_vec[lid] - b_vec[gid];\n" +
	        		"   partial_dot[lid] = diff*diff;\n" + 
	        		"\n" + 
	        		"   for(int i = group_size/2; i>0; i >>= 1) {\n" + 
	        		"      barrier(CLK_LOCAL_MEM_FENCE);\n" + 
	        		"      if(lid < i) {\n" + 
	        		"         partial_dot[lid] += partial_dot[lid + i];\n" + 
	        		"      }\n" + 
	        		"   }\n" + 
	        		"\n" + 
	        		"   if(lid == 0) {\n" + 
	        		"       sum = partial_dot[0]; \n" +
	        		"#if REAL_TYPE_SIZE == 4 \n" +
	        		"      output[grId] = sqrt( dot(sum, FOUR_ONES) ); \n" + 
	        		"#elif REAL_TYPE_SIZE == 8 \n" +
	        		"      output[grId] = sqrt( dot(sum.lo, FOUR_ONES) + dot(sum.hi, FOUR_ONES) ); \n" + 
	        		"#elif REAL_TYPE_SIZE == 16 \n" +
	        		"      output[grId] = sqrt( dot(sum.lo.lo, FOUR_ONES) + dot(sum.lo.hi, FOUR_ONES) + dot(sum.hi.lo, FOUR_ONES)  + dot(sum.hi.hi, FOUR_ONES) ); \n" +
	        		"#endif \n" +
	        		"   }\n" + 
	        		"}\n" +
	        		"\n" +
	        		"__kernel void dist2(__constant REAL_TYPE* a_vec, __global REAL_TYPE* b_vec,\n" + 
	        		"      __global float* output) {\n" + 
	        		"\n" + 
	        		"   int gid = get_global_id(0);\n" + 
	        		"   int group_size = get_local_size(0);\n" + 
	        		"   REAL_TYPE sum = 0.0f;\n" +
	        		"   REAL_TYPE c = 0.0f; \n" +
	        		"   for(int i = 0;i < group_size;i++) {\n" +
	        		"       // should be vector_size but since group_size == blockSize == vector_size/n and its float-n datatype, this works...\n" +
	        		"       REAL_TYPE diff = a_vec[i] - b_vec[i + gid*group_size]; \n" +
	        		"       diff = diff*diff;\n" +
	        		"       //sum += diff;\n" + 
					"       // Kahan Summation Algorithm aka compensated summation \n" +
					"       REAL_TYPE y = diff - c; \n" +
					"       REAL_TYPE t = sum + y; \n" +
					"       c = (t - sum) - y; \n" +
					"       sum = t; \n" +
	        		"   }\n" + 
	        		"\n" + 
	        		"#if REAL_TYPE_SIZE == 4 \n" +
	        		"      output[gid] = sqrt( dot(sum, FOUR_ONES) ); \n" + 
	        		"#elif REAL_TYPE_SIZE == 8 \n" +
	        		"      output[gid] = sqrt( dot(sum.lo, FOUR_ONES) + dot(sum.hi, FOUR_ONES) ); \n" + 
	        		"#elif REAL_TYPE_SIZE == 16 \n" +
	        		"      output[gid] = sqrt( dot(sum.lo.lo, FOUR_ONES) + dot(sum.lo.hi, FOUR_ONES) + dot(sum.hi.lo, FOUR_ONES)  + dot(sum.hi.hi, FOUR_ONES) ); \n" +
	        		"#endif \n" +
	        		"}"
	        		;
	        // for vector_size == 128, dist2 serial sum kernel performs better
		    // for vector_size == 512, dist reduce parallel kernel performs better
	        int vector_size = 512, num_vectors = 100*1024, deviceVectorData = 8;
	        dataSize = vector_size*num_vectors;
			blockSize = vector_size/deviceVectorData; // division by n to account for use of float-n
			
			CLProgram programDist = context.createProgram(distKernels);
            if(deviceVectorData == 4) {
            	programDist = programDist.defineMacro("REAL_TYPE", "float4").defineMacro("REAL_TYPE_SIZE", "4");
            }
            else if(deviceVectorData == 8) {
            	programDist = programDist.defineMacro("REAL_TYPE", "float8").defineMacro("REAL_TYPE_SIZE", "8");
            }
            else if(deviceVectorData == 16) {
            	programDist = programDist.defineMacro("REAL_TYPE", "float16").defineMacro("REAL_TYPE_SIZE", "16");
            }
            
			programDist = programDist.build();
			CLKernel kernelDist = programDist.createKernel("dist");
			CLKernel kernelDist2 = programDist.createKernel("dist2");
	        
			Random dataGenerator = new SecureRandom(new byte[] {0, 1, 2, 3});
			float[] a_vec = new float[vector_size];
			float[] b_vec = new float[dataSize];
			float[] output_vec = new float[num_vectors];
			
			for (int i = 0; i < vector_size; i++) {
	            a_vec[i] = dataGenerator.nextFloat();
	        }
			for (int i = 0; i < dataSize; i++) {
	            b_vec[i] = dataGenerator.nextFloat();
	        }
			
			for(int cnt = 0;cnt < 10;cnt++) { // performance test for host computation
				long t_startDist = System.currentTimeMillis();
				for(int i=0; i < num_vectors; i++) {
					float sum = 0.0f;
					float c = 0.0f;
					for(int j = 0;j < vector_size;j++) {
						float diff = a_vec[j] - b_vec[i*vector_size + j];
						diff = diff*diff; // input[i]
						//sum += diff;
						// Kahan Summation Algorithm aka compensated summation
						float y = diff - c;
						float t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}
					output_vec[i] = (float) Math.sqrt(sum);
				}
				long t_endDist = System.currentTimeMillis();
				System.out.println("Dist on host took " + (t_endDist - t_startDist) + "ms");
			}
			
			Pointer<Float> dist_result = executeOnDeviceDist(kernelDist, context, blockSize, num_vectors, a_vec, b_vec);
			float diff = 0.0f;
			for(int i = 0;i < num_vectors;i++) {
				float dist_res = dist_result.get(i);
				float dist_check = output_vec[i];
				diff += Math.abs(dist_check - dist_res);
			}
			System.out.println("Difference in results reduce kernel: " + diff);

			Pointer<Float> dist2_result = executeOnDeviceDist2(kernelDist2, context, blockSize, num_vectors, a_vec, b_vec);
			diff = 0.0f;
			for(int i = 0;i < num_vectors;i++) {
				float dist_res = dist2_result.get(i);
				float dist_check = output_vec[i];
				diff += Math.abs(dist_check - dist_res);
			}
			System.out.println("Difference in results serial reduce: " + diff);

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
