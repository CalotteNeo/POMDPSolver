package pomdp.utilities.datastructures;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

public abstract class Function implements Serializable {
	protected int[] m_aDims;
	protected int m_cDims;
	protected double m_dMinValue;
	protected double m_dMaxValue;
	protected double m_dDefaultValue;
	/**
	 * Transition Fucntion：
	 * aDims[0] = m_cStates;
	 * aDims[1] = m_cActions;
	 * aDims[2] = m_cStates;
	 * 
	 * O Function：
	 * aDims[0] = m_cActions;
	 * aDims[1] = m_cStates;
	 * aDims[2] = m_cObservations;
	 * @param aDims
	 */
	public Function( int[] aDims ){
		int iDim = 0;
		m_cDims = aDims.length;
		m_aDims = new int[m_cDims];
		for( iDim = 0 ; iDim < m_cDims ; iDim++ )
			m_aDims[iDim] = aDims[iDim];
		m_dMinValue = 0.0;
		m_dMaxValue = Double.MAX_VALUE * -1;
		m_dDefaultValue = 0.0;
	}
	
	public int getSize( int iDim ){
		return m_aDims[iDim];
	}
	
	public double getMaxValue(){
		return m_dMaxValue;
	}
	
	public double getMinValue(){
		return m_dMinValue;
	}
	
	//protected void setMaxValue( double dMaxValue ){
	//	m_dMaxValue = dMaxValue;
	//}
	
	//protected void setMinValue( double dMinValue ){
	//	m_dMinValue = dMinValue;
	//}
	
	public abstract double valueAt( int arg1 );
	public abstract double valueAt( int arg1, int arg2 );
	public abstract double valueAt( int arg1, int arg2, int arg3 );
	public abstract void setValue( int arg1, double dValue );
	public abstract void setValue( int arg1, int arg2, double dValue );
	
	/**
	 * 用于增加Transition Function的一个转换
	 * 参数：iStartState, iActionIdx, iEndState, dValue)
	 * @param arg1
	 * @param arg2
	 * @param arg3
	 * @param dValue
	 */
	public abstract void setValue( int arg1, int arg2, int arg3, double dValue );
	/**
	 *  获得和开始状态和动作有关的概率非0的转换
	 * @param arg1
	 * @param arg2
	 * @return
	 */
	public abstract Iterator<Entry<Integer,Double>> getNonZeroEntries( int arg1, int arg2 );
	public abstract Iterator getNonZeroEntries();
	public abstract int countNonZeroEntries( int arg1, int arg2 );

	protected void setAllValues( int iParam1, int iParam3, double dValue ){
		int iParam2 = 0;
		int cParam2 = m_aDims[1];
		for( iParam2 = 0 ;iParam2 < cParam2 ; iParam2++ ){
			setValue( iParam1, iParam2, iParam3, dValue );
		}
	}
	
	/**
	 * 用于增加O Function的一个观察值概率
	 * 最终均调用setValue( iParam1, iParam2, iParam3, dValue )
	 * 最终存入SparseTabularFunction.HashMap<Integer,Double>[][]
	 * 参数：iAction, iEndState, iObservation, dValue 
	 * 
	 * aDims[0] = m_cActions;
	 * aDims[1] = m_cStates;
	 * aDims[2] = m_cObservations;
	 * 
	 * @param iParam1
	 * @param iParam2
	 * @param iParam3
	 * @param dValue
	 */
	public void setAllValues( int iParam1, int iParam2, int iParam3, double dValue ){
		int cParam1 = m_aDims[0];//m_cActions
		//iAction, iEndState, iObservation都是任意
		if( iParam1 == -1 && iParam2 == -1 && iParam3 == -1 )
		{
			int cParam2 = m_aDims[1], cParam3 = m_aDims[2]; //m_cStates m_cObservations
			for( iParam1 = 0 ; iParam1 < cParam1 ; iParam1++ )
				for( iParam2 = 0 ; iParam2 < cParam2 ; iParam2++ )
					for( iParam3 = 0 ; iParam3 < cParam3 ; iParam3++ )
						setValue( iParam1, iParam2, iParam3, dValue );
			return;
		
		}
		//iAction任意
		if( iParam1 == -1 ){
			for( iParam1 = 0 ; iParam1 < cParam1 ; iParam1++ ){
				//iEndState任意
				if( iParam2 == -1 ){
					setAllValues( iParam1, iParam3, dValue );
				}
				else{
					setValue( iParam1, iParam2, iParam3, dValue );
				}
			}
		}
		else{
			//iEndState任意
			if( iParam2 == -1 ){
				setAllValues( iParam1, iParam3, dValue );
			}
			else{
				setValue( iParam1, iParam2, iParam3, dValue );
			}
		}
	}
	
	public void setValue( int[] parameters, double dValue ){
		switch( parameters.length ){
		case 1:
			setValue( parameters[0], dValue );
		case 2:
			setValue( parameters[0], parameters[1], dValue  );
		case 3:
			setValue( parameters[0], parameters[1], parameters[2], dValue );
		}
	}
	
	public double valueAt( int[] parameters ){
		switch( parameters.length ){
			case 0:
				return m_dDefaultValue;
			case 1:
				return valueAt( parameters[0] );
			case 2:
				return valueAt( parameters[0], parameters[1]  );
			case 3:
				return valueAt( parameters[0], parameters[1], parameters[2] );
		}
		return 0.0;
	}
	
	public abstract int countEntries();
	public abstract int countNonZeroEntries();

}
