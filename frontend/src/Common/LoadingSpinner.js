import React from 'react';

const LoadingSpinner = ({ size = "large", text = "Cargando...", className = "" }) => {
  const sizeClasses = {
    small: "h-4 w-4",
    medium: "h-8 w-8", 
    large: "h-12 w-12",
    xlarge: "h-16 w-16"
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-4 ${className}`}>
      <div className={`animate-spin rounded-full border-4 border-gray-600 border-t-blue-500 ${sizeClasses[size]}`}></div>
      {text && <p className="text-gray-400 text-center">{text}</p>}
    </div>
  );
};

export default LoadingSpinner;