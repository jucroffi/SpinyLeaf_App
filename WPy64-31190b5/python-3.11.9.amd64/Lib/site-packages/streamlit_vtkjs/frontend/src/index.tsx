import React from "react"
import ReactDOM from "react-dom"
import { ErrorBoundary } from "streamlit-component-lib-react-hooks"
import VTKStreamlit from "./VTKStreamlit"

import 'antd/dist/antd.css';

ReactDOM.render(
  <React.StrictMode>
    <ErrorBoundary>
      <VTKStreamlit />
    </ErrorBoundary>
  </React.StrictMode>,
  document.getElementById("root")
)
