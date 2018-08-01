package com.example.cnnlib;

import android.content.Context;
import android.opengl.GLES31;
import android.os.Environment;
import android.util.Log;

import com.example.cnnlib.eglenv.GLES31BackEnv;
import com.example.cnnlib.layer.ConvSSBO;
import com.example.cnnlib.layer.FullConnSSBO;
import com.example.cnnlib.layer.Input;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.layer.NonLinear;
import com.example.cnnlib.layer.Pooling;
import com.example.cnnlib.layer.SoftMax;
import com.example.cnnlib.utils.DataUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static com.example.cnnlib.utils.Constants.S_TEXTURE_SIZE;

/**
 * 网络只有8个缓存能够保存数据
 */
public class CnnNetwork {

    private static final String TAG = "CnnNetwork";

    private Context mContext;
    private ArrayList<Layer> mLayers;

    private volatile boolean mIsInit = false;
    private GLES31BackEnv mGLes31BackEnv;
    private String mNetStructureFile;
    private String mRootDir = Environment.getExternalStorageDirectory().getAbsolutePath();

    public CnnNetwork(Context context) {
        this.mContext = context;
        this.mLayers = new ArrayList<>();
        this.mGLes31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
    }

    public CnnNetwork(Context context, String netStructureFile) throws Exception {
        this.mContext = context;
        this.mLayers = new ArrayList<>();
        this.mGLes31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
        this.mNetStructureFile = netStructureFile;
        parse();
    }

    public void addLayer(Layer layer) {
        mLayers.add(layer);
    }

    public void initialize() {
        mGLes31BackEnv.post(this::actualInitialize);
    }

    public void predict(float[][][] input) {
        mGLes31BackEnv.post(() -> actualPredict(input));
    }


    private void actualPredict(float[][][] input) {
        runNet(input);
        actualReadOutput();
    }

    private void actualInitialize() {
        if (!mIsInit) {
            mIsInit = true;
            for (Layer layer : mLayers) {
                layer.initialize();
            }
        }
    }

    private void runNet(float[][][] input) {
        long begin = System.currentTimeMillis();
        int count = 1000;
        for (int ii = 0; ii < count; ii++) {

            long begin1 = System.currentTimeMillis();
            for (int i = 0; i < mLayers.size(); i++) {
                mLayers.get(i).forwardProc(input);
                GLES31.glFlush();
            }
            actualReadOutput();
            Log.w(TAG, "spent:" + (System.currentTimeMillis() - begin1));
        }
        Log.w(TAG, "ave spent:" + (System.currentTimeMillis() - begin) / count);
    }

    public void readOutput() {
        mGLes31BackEnv.post(this::actualReadOutput);
    }

    private void actualReadOutput() {
        Layer layer = mLayers.get(mLayers.size() - 1);
        DataUtils.readOutput(layer);
    }

    private void parse() throws Exception {

        File f = new File(mNetStructureFile);
        Scanner s = new Scanner(f);

        int layerNum = 0;
        Layer preLayer = null;
        while (s.hasNextLine()) {
            String str = s.nextLine();
            str = str.trim();
            String strLow = str.toLowerCase();

            if (strLow.startsWith("root_directory")) {
                str = str.substring(14);
                str = deriveStr(str);
                if (str.equals("")) {
                    throw new Exception("root_directory 为空.");
                }
                mRootDir = str;
            } else if (strLow.startsWith("layer")) {
                str = str.substring(5);
                String tempStr;
                while (s.hasNextLine()) {
                    str += "\n";
                    tempStr = s.nextLine();
                    str += tempStr;
                    if (tempStr.contains("}"))
                        break;
                }
                Layer layer = deriveLayer(str, preLayer);
                if (layer == null) {
                    throw new Exception("第" + layerNum + "层解析出错.");
                } else {
                    preLayer = layer;
                }
                layerNum++;
            } else if (strLow.equals(""))
                continue;
            else {
                throw new Exception("网络文件格式错误: " + str);
            }
        }

    }


    private String deriveStr(String str) {
        str = str.trim();
        if (!str.startsWith(":"))
            return "";
        str = str.substring(1);

        str = str.trim();
        if (!str.startsWith("\""))
            return "";
        str = str.substring(1);

        int i = str.indexOf("\"");
        if (i == -1)
            return "";

        String retStr = str.substring(0, i);

        if (i != str.length() - 1) {
            str = str.substring(i);
            str = str.trim();
            if (!str.equals(""))
                return "";
        }

        return retStr;
    }

    private String deriveNum(String str) {
        str = str.trim();
        if (!str.startsWith(":"))
            return "";
        str = str.substring(1);

        str = str.trim();
        int i = str.indexOf(" ");

        if (i == -1)
            return str;

        if (!str.substring(i).trim().equals(""))
            return "";
        else
            return str.substring(0, i);
    }


    private Layer deriveLayer(String str, Layer preLayer) {
        String[] strArr = str.split("\n");

        strArr[0] = strArr[0].trim();
        strArr[1] = strArr[1].trim();

        if (strArr[0].startsWith("{"))
            strArr[0] = strArr[0].substring(1);
        else if (strArr[1].startsWith("{"))
            strArr[1] = strArr[1].substring(1);
        else
            return null;

        String endStr = strArr[strArr.length - 1];
        endStr = endStr.trim();
        if (!endStr.startsWith("}"))
            return null;
        endStr = endStr.substring(1);
        endStr = endStr.trim();
        if (!endStr.equals(""))
            return null;

        String type = "";
        String name = "";
        List<String> args = new ArrayList<String>();
        List<String> values = new ArrayList<String>();

        for (int i = 0; i < strArr.length - 1; ++i) {
            strArr[i] = strArr[i].trim();
            if (strArr[i].equals(""))
                continue;

            int i1 = strArr[i].indexOf(":");
            int i2 = strArr[i].indexOf(" ");
            int j = (i1 < i2) ? i1 : i2;
            String tempArg = strArr[i].substring(0, j);
            String tempValue;
            String temp = strArr[i].substring(j);
            if (temp.contains("\""))
                tempValue = deriveStr(temp);
            else
                tempValue = deriveNum(temp);

            if (tempArg.equalsIgnoreCase("name"))
                name = tempValue;
            else if (tempArg.equalsIgnoreCase("type"))
                type = tempValue;
            else {
                args.add(tempArg);
                values.add(tempValue);
            }
        }

        if (type.equalsIgnoreCase("Input")) {
            int width = -1;
            int height = -1;
            int channel = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("width")) {
                    width = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("height")) {
                    height = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("channel")) {
                    channel = Integer.parseInt(tempValue);
                } else {
                    return null;
                }
            }
            if (width == -1 || height == -1 || channel == -1) {
                return null;
            } else {
                Input in = new Input(mContext, new int[]{width, height, channel});
                mLayers.add(in);
                return in;
            }
        } else if (type.equalsIgnoreCase("Convolution")) {
            String parametersFile = null;
            int pad = -1;
            int stride = -1;
            int kennel_amount = -1;
            int kennel_width = -1;
            int kennel_height = -1;
            int kennel_channel = -1;
            NonLinear.NonLinearType nonLinearType = NonLinear.NonLinearType.NONE;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("parameters_file")) {
                    parametersFile = tempValue;
                } else if (tempArg.equalsIgnoreCase("pad")) {
                    pad = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("stride")) {
                    stride = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("kennel_amount")) {
                    kennel_amount = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("kennel_width")) {
                    kennel_width = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("kennel_height")) {
                    kennel_height = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("kennel_channel")) {
                    kennel_channel = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("active")) {
                    if ("ReLU".equalsIgnoreCase(tempValue)) {
                        nonLinearType = NonLinear.NonLinearType.RELU;
                    }
                } else {
                    return null;
                }
            }
            if (parametersFile == null || pad == -1 || stride == -1 || kennel_amount == -1 ||
                    kennel_width == -1 || kennel_height == -1 || kennel_channel == -1) {
                return null;
            } else {
                ConvSSBO c = new ConvSSBO(mContext, preLayer, kennel_amount, new int[]{kennel_width, kennel_height, kennel_channel}, pad, new int[]{stride, stride},
                        nonLinearType, mRootDir + parametersFile);
                mLayers.add(c);
                return c;
            }
        } else if (type.equalsIgnoreCase("Pooling")) {
            String pool = null;
            int kernelSize = -1;
            int pad = -1;
            int stride = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("pool")) {
                    pool = tempValue;
                } else if (tempArg.equalsIgnoreCase("kernel_size")) {
                    kernelSize = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("pad")) {
                    pad = Integer.parseInt(tempValue);
                } else if (tempArg.equalsIgnoreCase("stride")) {
                    stride = Integer.parseInt(tempValue);
                } else {
                    return null;
                }
            }
            if (pool == null || pad == -1 || stride == -1 || kernelSize == -1) {
                return null;
            } else {
                Pooling p = new Pooling(mContext, preLayer, new int[]{kernelSize, kernelSize},
                        new int[]{stride, stride});
                mLayers.add(p);
                return p;
            }
        } else if (type.equalsIgnoreCase("FullyConnected")) {
            String parametersFile = null;
            int kennel_amount = -1;
            NonLinear.NonLinearType nonLinearType = NonLinear.NonLinearType.NONE;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("parameters_file")) {
                    parametersFile = tempValue;
                } else if (tempArg.equalsIgnoreCase("kennel_amount")) {
                    kennel_amount = Integer.parseInt(tempValue);

                } else if (tempArg.equalsIgnoreCase("active")) {
                    if ("ReLU".equalsIgnoreCase(tempValue)) {
                        nonLinearType = NonLinear.NonLinearType.RELU;
                    } else {
                        return null;
                    }
                }
            }
            if (parametersFile == null || kennel_amount == -1) {
                return null;
            } else {
                FullConnSSBO fc = new FullConnSSBO(mContext, preLayer, kennel_amount, nonLinearType, mRootDir + parametersFile);
                mLayers.add(fc);
                return fc;
            }
        } else if (type.equalsIgnoreCase("Softmax")) {
            int amount = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("amount"))
                    amount = Integer.parseInt(tempValue);

            }
            if (amount == -1) {
                return null;
            } else {
                SoftMax sm = new SoftMax(mContext, preLayer, amount);
                mLayers.add(sm);
                return sm;
            }
        } else
            return null;
    }


}
