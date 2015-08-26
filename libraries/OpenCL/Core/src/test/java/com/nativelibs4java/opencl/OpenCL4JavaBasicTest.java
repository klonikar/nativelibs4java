package trial.javacl;

import static com.nativelibs4java.opencl.JavaCL.createBestContext;
import static org.bridj.Pointer.pointerToFloats;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
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

// javac -cp javacl-1.0.0-RC3-shaded.jar -d . OpenCL4JavaBasicTest.java
// java -cp javacl-1.0.0-RC3-shaded.jar;. trial.javacl.OpenCL4JavaBasicTest 1000000
public class OpenCL4JavaBasicTest {
	private static class MyRunnable implements Runnable {
		private int start;
		private int end;
		public float[] expectedResults;
		public int chunkSize;

		public MyRunnable(int start, int end) {
			this.start = start;
			this.end = end;
			this.chunkSize = end - start;
			int dataSize = end - start + 1;
			expectedResults = new float[dataSize];
		}
		
		public void run() {
			for (int i = start; i < end; i++) {
				float a = i;
				float b = i;
                expectedResults[i-start] = (float) Math.sin(Math.exp(Math.cos(Math.sin(a) * Math.sin(b) + 1)));
			}
		}
	}

    private static MyRunnable[] executeOnHost(int dataSize) {
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
		System.out.println("Host execute time: " + (t2-t1) + " ms");
		return tasks;
    }

	private static class MyRunnableDouble implements Runnable {
		private int start;
		private int end;
		public double[] expectedResults;
		public int chunkSize;

		public MyRunnableDouble(int start, int end) {
			this.start = start;
			this.end = end;
			this.chunkSize = end - start;
			int dataSize = end - start + 1;
			expectedResults = new double[dataSize];
		}
		
		public void run() {
			for (int i = start; i < end; i++) {
				float a = i;
				float b = i;
                expectedResults[i-start] = Math.sin(Math.exp(Math.cos(Math.sin(a) * Math.sin(b) + 1)));
			}
		}
	}

    private static MyRunnableDouble[] executeOnHostDouble(int dataSize) {
		int numProcessors = Runtime.getRuntime().availableProcessors();
		int chunkSize = dataSize/numProcessors;
		System.out.println("number of processors/cores: " + numProcessors + ", CPU chunkSize: " + chunkSize);
		long t1 = System.currentTimeMillis();
		ExecutorService taskExecutor = Executors.newFixedThreadPool(numProcessors);
		MyRunnableDouble[] tasks = new MyRunnableDouble[numProcessors];
		for(int i = 0;i < numProcessors;i++) {
			MyRunnableDouble task = new MyRunnableDouble(i*chunkSize, (i+1)*chunkSize);
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
        double avgAbsoluteError = totalAbsoluteError / dataSize;
        double avgRelativeError = totalRelativeError / dataSize;
        return new double[] {avgAbsoluteError, avgRelativeError};
    }

    private static Pointer<Double> executeOnDeviceDouble(CLKernel kernel, CLContext context, int dataSize, int blockSize) {
        CLQueue queue = context.createDefaultQueue();

        double[] aVals = new double[dataSize];
        double[] bVals = new double[dataSize];
        for (int i = 0; i < dataSize; i++) {
            double value = (double)i;
            //a.set(i, value);
            //b.set(i, value);
            aVals[i] = value;
            bVals[i] = value;
        }
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
		System.out.println("Device data transfer time: " + ((t_dataXfr1_g - t1_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr1_g) + "ms");
		System.out.println("Device time diff: " + (t2_g-t1_g) + " ms");

		return output;
    }

    private static Pointer<Float> executeOnDevice(CLKernel kernel, CLContext context, int dataSize, int blockSize) {
        CLQueue queue = context.createDefaultQueue();
        float[] aVals = new float[dataSize];
        float[] bVals = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            float value = (float)i;
            //a.set(i, value);
            //b.set(i, value);
            aVals[i] = value;
            bVals[i] = value;
        }
        // Ask for execution of the kernel with global size = dataSize
        int numThreads = ((dataSize-1)/blockSize + 1)*blockSize;
		System.out.println("dataSize: " + dataSize + ", numThreads: " + numThreads + ", blockSize: " + blockSize);

		long t1_g = System.currentTimeMillis();
        /// Create direct NIO buffers and fill them with data in the correct byte order
        Pointer<Float> a = pointerToFloats(aVals).order(context.getKernelsDefaultByteOrder());
        Pointer<Float> b = pointerToFloats(bVals).order(context.getKernelsDefaultByteOrder());

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
		System.out.println("Device data transfer time: " + ((t_dataXfr1_g - t1_g) + (t2_g - t_execute_g)) + "ms, execute time: " + (t_execute_g - t_dataXfr1_g) + "ms");
		System.out.println("Device time diff: " + (t2_g-t1_g) + " ms");

		return output;
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
				Pointer<Double> output = executeOnDeviceDouble(kernel, context, dataSize, blockSize);
				MyRunnableDouble[] tasks = executeOnHostDouble(dataSize);
				double[] diff = computeDifferenceDouble(output, tasks,  dataSize);
	            System.out.println("Average absolute error = " + diff[0]);
	            System.out.println("Average relative error = " + diff[1]);				
			}
			else {
				Pointer<Float> output = executeOnDevice(kernel, context, dataSize, blockSize);
				MyRunnable[] tasks = executeOnHost(dataSize);
				double[] diff = computeDifference(output, tasks,  dataSize);
	            System.out.println("Average absolute error = " + diff[0]);
	            System.out.println("Average relative error = " + diff[1]);
			}

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}