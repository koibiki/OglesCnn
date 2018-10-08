package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.MessagePackUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_FULL_CONN_SHADER_HEADER;


/**
 * 全连接层 视为 1, 1, amount的特殊image
 * 实际存储为一张texture w h depth 为  1, 1, ceil(amount/4)
 */
public class FullConnSSBO extends Layer {

    private static final String TAG = "FullConnSSBO";

    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mBindLayer;         // 纹理绑定深度编号

    private float[][] mKernels;
    private int mShaderPro;
    private int mKernelAmount;

    private int[] mParams;

    private ActiveType mType;
    private int[] mKernelBuffer = new int[1];
    private int mNumGroupX;

    public FullConnSSBO(Context context, String name, Layer preLayer, int kernelAmount, ActiveType type) {
        super(context, name, preLayer);
        this.mKernelAmount = kernelAmount;
        this.mType = type;
        this.mOutShape = calculateFullShape(kernelAmount);
        this.mOut = FloatBuffer.allocate(mOutShape[0] * mOutShape[1] * 4);
        this.mBindLayer = 0;
//        this.mBindLayer = Utils.alignBy4(mOutShape[2]) / 4;
    }

    private int[] calculateFullShape(int kennelAmount) {
        return new int[]{1, 1, kennelAmount};
    }

    public int getKernelAmount() {
        return mKernelAmount;
    }

    @Override
    public void initialize() {
        initFullConnect();
    }

    private void initFullConnect() {
        int kennelSize = mInShape[0] * mInShape[1] * Utils.alignBy4(mInShape[2]) + 4;

        int xSize;
        if (Utils.alignBy4(mKernelAmount) / 4 <= GLES31BackEnv.getMaxWorkGroupSize()) {
            xSize = Utils.alignBy4(mKernelAmount) / 4;
            mNumGroupX = 1;
        } else {
            xSize = GLES31BackEnv.getMaxWorkGroupSize();
            mNumGroupX = (int) Math.ceil(Utils.alignBy4(mKernelAmount) * 1.0f / 4 / GLES31BackEnv.getMaxWorkGroupSize());
        }

        String source = createShaderSource(xSize, kennelSize);
        mShaderPro = initCompPro(source);

        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(1, 1, Utils.alignBy4(mKernelAmount) / 4);

        int kennelBufSize = kennelSize * mKernelAmount;

//        mKernels = createKennels(kennelSize);
        mKernels = loadKernels(kennelSize);

        mKernelBuffer[0] = Render.initKernelBuffer(kennelBufSize);
        transferKennelToBuffer(mKernels);

        createShaderParams();
    }

    private String createShaderSource(int xSize, int kernelSize) {
        String shaderFile = "full_connect.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_FULL_CONN_SHADER_HEADER, kernelSize, mKernelAmount, xSize, 1, 1) + source;
    }

    private void createShaderParams() {
        mParams = new int[9];
        mParams[0] = mInShape[0];
        mParams[1] = mInShape[1];
        mParams[2] = mInShape[2];
        mParams[3] = mOutShape[0];
        mParams[4] = mOutShape[1];
        mParams[5] = mOutShape[2];
        mParams[6] = Utils.alignBy4(mInShape[2]) / 4;
        mParams[7] = Utils.alignBy4(mOutShape[2]) / 4;
        mParams[8] = mType.index;
    }

    private float[][] loadKernels(int kernelSize) {
        String kernel_file = mName;
        Object[] objects = MessagePackUtils.unpackParam(mContext, kernel_file, new Class[]{float[].class, float[].class});
        if (objects != null) {
            float[] kernel = (float[]) objects[0];
            float[] bias = (float[]) objects[1];
            return transferKernelFormat(kernel, bias, kernelSize);
        } else {
            return null;
        }
    }

    /**
     * 按面放置weight, 每方满一面后, 再放置下一个channel的weight面
     * 输入的kernel
     */
    private float[][] transferKernelFormat(float[] kernels, float[] bias, int kernelSize) {
        float[][] new_kernels = new float[mKernelAmount][kernelSize];

        for (int a = 0; a < mKernelAmount; a++) {
            for (int w = 0; w < mInShape[0]; w++) {
                for (int h = 0; h < mInShape[1]; h++) {
                    for (int c = 0; c < mInShape[2]; c++) {
                        new_kernels[a][c * mInShape[0] * mInShape[1] + h * mInShape[0] + w] = kernels[a * mInShape[0] * mInShape[1] * mInShape[2] + c * mInShape[0] * mInShape[1] + h * mInShape[0] + w];
                    }
                }
            }
            new_kernels[a][kernelSize - 4] = bias[a];
        }
        return new_kernels;
    }

    private void transferKennelToBuffer(float[][] kernels) {
        for (int a = 0; a < mKernelAmount; a++) {
            float[] kennel = kernels[a];
            int offset = a * kennel.length;
            Render.transferToBuffer(FloatBuffer.wrap(kennel), mKernelBuffer[0], offset);
        }
    }

    private float[][] createKennels(int kernelSize) {
        return TestDataCreator.createFullConnKennel(kernelSize, mKernelAmount, mInShape);
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArrayAndBuffer(mOutTex, mAttachID, mBindLayer, mKernelBuffer[0]);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performFullConnectSSBO(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKernelBuffer[0], mNumGroupX);
    }

    @Override
    public float[][][] readResult() {
        float[][][] out = new float[1][1][mKernelAmount];
        DataUtils.readOutput(this, mOut);
        for (int i = 0; i < mKernelAmount; i++) {
            out[0][0][i] = mOut.get(i);
        }
        return out;
    }

}
